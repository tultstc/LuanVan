"""CPU self-check for the FGTN-AVA modules (no mmaction / GPU needed).

    python projects/fgtn_ava/selfcheck.py

Checks: single- and multi-level neck shapes, every branch_mode, old-checkpoint
key compatibility, and that weighted_focal_bce reduces to plain BCE when
gamma=0 / alpha=1 / class_weight=None.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from projects.fgtn_ava.fgtn_neck import FGTNNeckAVA  # noqa: E402
from projects.fgtn_ava.heads import weighted_focal_bce  # noqa: E402
from projects.fgtn_ava.make_class_weight import class_weights  # noqa: E402

torch.manual_seed(0)

# single level, all branch modes
x4 = torch.randn(1, 768, 16, 7, 7)
for mode in ("hybrid", "mamba_only", "attn_only", "no_gate"):
    assert FGTNNeckAVA(768, num_blocks=1, branch_mode=mode)(x4).shape == x4.shape, mode

# multi level: tuple in, tuple out, shapes preserved per level
x3 = torch.randn(1, 384, 16, 14, 14)
neck = FGTNNeckAVA((384, 768), num_blocks=2)
y3, y4 = neck((x3, x4))
assert y3.shape == x3.shape and y4.shape == x4.shape
assert len(neck.level_blocks) == 2 and len(neck.level_blocks[0]) == 2
assert neck.level_blocks[0][0].norm.weight.shape[0] == 384 and neck.level_blocks[1][0].norm.weight.shape[0] == 768

# old single-level checkpoints (neck.blocks.*) still load
old = {k.replace("level_blocks.0.", "blocks.", 1): v for k, v in FGTNNeckAVA(768, num_blocks=2).state_dict().items()}
assert all(k.startswith("blocks.") for k in old)
FGTNNeckAVA(768, num_blocks=2).load_state_dict(old)

# loss: default args == plain BCE; weights/focal change it as expected
logits, labels = torch.randn(6, 5), (torch.rand(6, 5) > 0.5).float()
assert torch.allclose(weighted_focal_bce(logits, labels), F.binary_cross_entropy_with_logits(logits, labels))
assert weighted_focal_bce(logits, labels, gamma=2.0) < weighted_focal_bce(logits, labels)  # easy terms down-weighted
assert torch.allclose(weighted_focal_bce(logits, labels, class_weight=[2.0] * 5), 2 * weighted_focal_bce(logits, labels))

# class weights: rarer class -> larger weight, mean 1, clipped
w = class_weights({1: 1000, 2: 10, 3: 0}, num_classes=3, beta=0.999, clip=10.0)
assert w[0] < w[1] < w[2] and max(w) <= 10.0

print("selfcheck ok")
