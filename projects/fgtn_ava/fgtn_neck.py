"""Hybrid FGTN-Mamba neck for MMAction2 spatio-temporal detection (AVA).

Drop-in temporal neck between the video backbone and the RoI head. For each
spatial position of the 5D feature map (B, C, T, H, W), two parallel branches
reason along the temporal axis T and are fused by a learned per-channel gate:
    - global branch: bidirectional selective SSM (Mamba), O(TC)
    - local branch: windowed MHSA (W=5, 4 heads), O(TWC)
    out = g * x_mamba + (1 - g) * x_local,  g = sigmoid(Conv1d(x))

Self-contained on purpose (no OpenTAD imports) so it can be dropped into any
MMAction2 project: `custom_imports = dict(imports=["projects.fgtn_ava.fgtn_neck"])`.

branch_mode enables the ablation table from a single module:
    "hybrid" | "mamba_only" | "attn_only" | "no_gate"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba as ViM

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class FGTNNeckAVA(nn.Module):
    def __init__(
        self,
        in_channels,
        num_blocks=2,
        branch_mode="hybrid",
        n_head=4,
        window_size=5,
        mamba_kernel_size=4,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            HybridFGTNBlock(
                in_channels,
                branch_mode=branch_mode,
                n_head=n_head,
                window_size=window_size,
                mamba_kernel_size=mamba_kernel_size,
                drop_path_rate=drop_path_rate,
            )
            for _ in range(num_blocks)
        )

    def _forward_single(self, feat):
        # feat: (B, C, T, H, W) -> temporal sequences per spatial position
        B, C, T, H, W = feat.shape
        x = feat.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        for block in self.blocks:
            x = block(x)
        return x.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, x):
        # mmdet detectors pass backbone outputs as a tensor or a tuple of tensors
        if isinstance(x, (list, tuple)):
            return type(x)(self._forward_single(f) for f in x)
        return self._forward_single(x)


class HybridFGTNBlock(nn.Module):
    """Pre-norm residual block: bidirectional Mamba || windowed MHSA, sigmoid-gated fusion."""

    def __init__(
        self,
        n_embd,
        branch_mode="hybrid",
        n_head=4,
        window_size=5,
        mamba_kernel_size=4,
        drop_path_rate=0.1,
    ):
        super().__init__()
        assert branch_mode in ("hybrid", "mamba_only", "attn_only", "no_gate")
        self.branch_mode = branch_mode

        self.norm = nn.LayerNorm(n_embd, eps=1e-6)

        if branch_mode != "attn_only":
            if MAMBA_AVAILABLE:
                self.mamba = ViM(n_embd, d_conv=mamba_kernel_size, bimamba_type="v2", use_fast_path=True)
            else:
                # ponytail: pure-pytorch O(T)-loop scan, CPU dev only; use mamba-ssm CUDA kernel for training
                self.mamba = NaiveBiMamba(n_embd, d_conv=mamba_kernel_size)
        if branch_mode != "mamba_only":
            self.attn = WindowedMHSA(n_embd, n_head=n_head, window_size=window_size)
        if branch_mode == "hybrid":
            self.gate = nn.Conv1d(n_embd, n_embd, kernel_size=1)

        self.drop_path_rate = drop_path_rate

    def _drop_path(self, x):
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = 1.0 - self.drop_path_rate
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x):
        # x: (B, C, T)
        res = x
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)

        if self.branch_mode != "attn_only":
            x_mamba = self.mamba(x_norm.transpose(1, 2)).transpose(1, 2)
        if self.branch_mode != "mamba_only":
            x_attn = self.attn(x_norm)

        if self.branch_mode == "hybrid":
            g = torch.sigmoid(self.gate(x_norm))  # (B, C, T), per-channel routing
            out = g * x_mamba + (1.0 - g) * x_attn
        elif self.branch_mode == "no_gate":
            out = 0.5 * x_mamba + 0.5 * x_attn
        elif self.branch_mode == "mamba_only":
            out = x_mamba
        else:  # attn_only
            out = x_attn

        return res + self._drop_path(out)


class WindowedMHSA(nn.Module):
    """MHSA restricted to a local temporal window via a band mask.

    ponytail: full T x T attention with a band mask -- at the neck T is ~16,
    so O(T^2) is negligible; swap for chunked attention if T grows past ~512.
    """

    def __init__(self, n_embd, n_head=4, window_size=5):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.half_window = window_size // 2
        self.qkv = nn.Linear(n_embd, n_embd * 3)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        qkv = self.qkv(x.transpose(1, 2))  # (B, T, 3C)
        q, k, v = qkv.reshape(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)

        idx = torch.arange(T, device=x.device)
        band = (idx[None, :] - idx[:, None]).abs() <= self.half_window  # (T, T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=band)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out).transpose(1, 2)


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


try:
    from mmaction.registry import MODELS

    MODELS.register_module(module=FGTNNeckAVA)
except ImportError:
    pass  # standalone use (e.g. CPU self-check) without mmaction installed
