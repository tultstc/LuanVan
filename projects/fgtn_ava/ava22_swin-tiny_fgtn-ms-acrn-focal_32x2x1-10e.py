# Full pipeline: multi-scale Hybrid FGTN-Mamba (Swin stages 3+4) + ACRN + focal/class-weighted loss.
# Every other setting (data, schedule, optimizer) is inherited from the single-scale recipe
# so that the pair {base config, this config} isolates the three additions.
# Run from the repo root (custom_imports resolves projects.fgtn_ava.*).

_base_ = ["./ava22_swin-tiny_fgtn_32x2x1-10e.py"]

custom_imports = dict(
    imports=["projects.fgtn_ava.fgtn_neck", "projects.fgtn_ava.heads"], allow_failed_imports=False
)

# Swin-T layer indices are 0-based: 2 -> stage 3 (C=384, stride 16), 3 -> stage 4 (C=768, stride 32).
# Optional: json list from make_class_weight.py (None = uniform weights, focal only).
class_weight = None

model = dict(
    backbone=dict(out_indices=(2, 3)),
    # one Hybrid FGTN-Mamba stack per level, no parameter sharing across levels
    neck=dict(in_channels=(384, 768)),
    roi_head=dict(
        # SingleRoIExtractor3D resizes every level to the finest map (stride 16) and concatenates
        # along channels -> 384 + 768 = 1152 channels before RoIAlign
        bbox_roi_extractor=dict(featmap_stride=16),
        # ACRN: tile the pooled actor vector over the scene map and learn actor-scene relations
        # (in_channels = actor 1152 + scene 1152)
        shared_head=dict(type="ACRNHead", in_channels=2304, out_channels=1152),
        bbox_head=dict(
            type="BBoxHeadAVAWeighted",
            in_channels=1152,
            focal_gamma=2.0,
            focal_alpha=1.0,
            class_weight=class_weight,
        ),
    ),
)
