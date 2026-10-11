"""Shape/gradient self-check for FGTNProj. Runs on CPU (naive Mamba fallback).

Usage: python tools/fgtn_selfcheck.py

Bootstraps a minimal package context so the check runs without the full
training stack (mmcv/mmaction/compiled ops): only the registry in
`opentad.models.builder` is stubbed; bricks and fgtn_proj are the real files.
"""

import importlib
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap():
    from mmengine.registry import Registry

    for name in ("opentad", "opentad.models", "opentad.models.projections"):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(REPO_ROOT / name.replace(".", "/"))]
        sys.modules[name] = pkg

    builder = types.ModuleType("opentad.models.builder")
    builder.MODELS = Registry("models")
    builder.PROJECTIONS = builder.MODELS
    sys.modules["opentad.models.builder"] = builder

    importlib.import_module("opentad.models.bricks")
    return importlib.import_module("opentad.models.projections.fgtn_proj")


fgtn = _bootstrap()


def check_mode(branch_mode):
    B, C_in, C_out, T = 2, 768, 256, 64
    model = fgtn.FGTNProj(
        in_channels=C_in,
        out_channels=C_out,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        branch_mode=branch_mode,
    )
    x = torch.randn(B, C_in, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, T // 2 :] = False  # padded sample

    feats, masks = model(x, mask)

    assert len(feats) == 6, f"{branch_mode}: expected 6 pyramid levels, got {len(feats)}"
    t = T
    for i, (f, m) in enumerate(zip(feats, masks)):
        assert f.shape == (B, C_out, t), f"{branch_mode} level {i}: {f.shape} != {(B, C_out, t)}"
        assert m.shape == (B, t) and m.dtype == torch.bool
        assert torch.isfinite(f).all(), f"{branch_mode} level {i}: non-finite output"
        t //= 2

    # gradients flow through the block (both branches + gate in hybrid mode)
    loss = sum(f.sum() for f in feats)
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grads > 0, f"{branch_mode}: no gradients"
    print(f"  [ok] {branch_mode}: 6 levels, grads on {n_grads} params")


if __name__ == "__main__":
    print(f"mamba-ssm available: {fgtn.MAMBA_AVAILABLE} (False -> naive CPU fallback)")
    for mode in ("hybrid", "mamba_only", "attn_only", "no_gate"):
        check_mode(mode)
    print("all checks passed")
