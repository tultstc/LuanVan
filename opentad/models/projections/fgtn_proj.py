import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actionformer_proj import get_sinusoid_encoding
from ..bricks import ConvModule, AffineDropPath
from ..bricks.transformer import LocalMaskedMHCA
from ..builder import PROJECTIONS

try:
    from mamba_ssm.modules.mamba_simple import Mamba as ViM

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


@PROJECTIONS.register_module()
class FGTNProj(nn.Module):
    """Hybrid FGTN-Mamba temporal neck.

    Two parallel branches along the temporal axis, fused by a learned
    per-channel sigmoid gate:
        - global branch: bidirectional selective SSM (Mamba), O(TC)
        - local branch: windowed MHSA (W=5, 4 heads), O(TWC)
        out = g * x_mamba + (1 - g) * x_local,  g = sigmoid(Conv1d(x))

    branch_mode enables the ablation table from a single module:
        "hybrid"     - both branches + learned gate (default)
        "mamba_only" - global SSM branch only (S3TN baseline)
        "attn_only"  - local windowed MHSA branch only
        "no_gate"    - both branches, fixed 0.5/0.5 fusion
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem blocks, #branch blocks)
        conv_cfg=None,  # kernel_size
        norm_cfg=None,
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
        branch_mode="hybrid",
        attn_cfg=dict(n_head=4, window_size=5, attn_pdrop=0.0, proj_pdrop=0.0),
        mamba_cfg=dict(kernel_size=4, drop_path_rate=0.3),
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(HybridFGTNBlock(out_channels, branch_mode=branch_mode, attn_cfg=attn_cfg, **mamba_cfg))

        # main branch with downsampling
        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(
                HybridFGTNBlock(out_channels, n_ds_stride=2, branch_mode=branch_mode, attn_cfg=attn_cfg, **mamba_cfg)
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem blocks
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


class HybridFGTNBlock(nn.Module):
    """Pre-norm residual block: bidirectional Mamba || windowed MHSA, sigmoid-gated fusion."""

    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=4,  # mamba conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        drop_path_rate=0.3,  # drop path rate
        branch_mode="hybrid",
        attn_cfg=dict(n_head=4, window_size=5, attn_pdrop=0.0, proj_pdrop=0.0),
    ):
        super().__init__()
        assert branch_mode in ("hybrid", "mamba_only", "attn_only", "no_gate")
        self.branch_mode = branch_mode

        self.norm = nn.LayerNorm(n_embd, eps=1e-6)

        if branch_mode != "attn_only":
            if MAMBA_AVAILABLE:
                self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
            else:
                # ponytail: pure-pytorch O(T)-loop scan, CPU dev only; use mamba-ssm CUDA kernel for training
                self.mamba = NaiveBiMamba(n_embd, d_conv=kernel_size)
        if branch_mode != "mamba_only":
            self.attn = LocalMaskedMHCA(
                n_embd,
                attn_cfg["n_head"],
                window_size=attn_cfg["window_size"],
                attn_pdrop=attn_cfg.get("attn_pdrop", 0.0),
                proj_pdrop=attn_cfg.get("proj_pdrop", 0.0),
            )
        if branch_mode == "hybrid":
            self.gate = nn.Conv1d(n_embd, n_embd, kernel_size=1)

        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)  # (B, C, T)

        if self.branch_mode != "attn_only":
            x_mamba = self.mamba(x_norm.transpose(1, 2)).transpose(1, 2)
        if self.branch_mode != "mamba_only":
            x_attn, _ = self.attn(x_norm, mask)

        if self.branch_mode == "hybrid":
            g = torch.sigmoid(self.gate(x_norm))  # (B, C, T), per-channel routing
            out = g * x_mamba + (1.0 - g) * x_attn
        elif self.branch_mode == "no_gate":
            out = 0.5 * x_mamba + 0.5 * x_attn
        elif self.branch_mode == "mamba_only":
            out = x_mamba
        else:  # attn_only
            out = x_attn

        out = out * mask.unsqueeze(1).to(x.dtype)
        x = res + self.drop_path(out)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return x, mask


class MaxPooler(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.stride = stride

    def forward(self, x, mask, **kwargs):
        if self.stride > 1:
            out_mask = self.ds_pooling(mask.float()).bool()
        else:
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.unsqueeze(1).to(x.dtype)
        return out, out_mask.bool()


class NaiveSelectiveSSM(nn.Module):
    """Reference (slow) selective SSM: h_t = exp(dt*A) h_{t-1} + dt*B x_t, y_t = C h_t + D x_t.

    Same recurrence as mamba-ssm's selective_scan, implemented as a plain python
    loop over T so it runs on CPU. Numerically equivalent, ~100x slower.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        B, T, _ = x.shape
        d_state = self.A_log.shape[1]

        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        # causal depthwise conv along T
        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_in = F.silu(x_in)

        dbc = self.x_proj(x_in)  # (B, T, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(dbc, [self.dt_rank, d_state, d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, T, d_inner, d_state)
        dBx = dt.unsqueeze(-1) * B_ssm.unsqueeze(2) * x_in.unsqueeze(-1)  # (B, T, d_inner, d_state)

        h = x.new_zeros(B, self.d_inner, d_state)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bds,bs->bd", h, C_ssm[:, t]))
        y = torch.stack(ys, dim=1) + self.D * x_in  # (B, T, d_inner)

        y = y * F.silu(z)
        return self.out_proj(y)


class NaiveBiMamba(nn.Module):
    """Bidirectional wrapper: forward scan + backward scan (flipped), averaged."""

    def __init__(self, d_model, d_conv=4, **kwargs):
        super().__init__()
        self.fwd = NaiveSelectiveSSM(d_model, d_conv=d_conv)
        self.bwd = NaiveSelectiveSSM(d_model, d_conv=d_conv)

    def forward(self, x):
        # x: (B, T, C)
        return 0.5 * (self.fwd(x) + self.bwd(x.flip(1)).flip(1))
