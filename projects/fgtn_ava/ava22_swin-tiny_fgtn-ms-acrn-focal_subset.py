# Full pipeline (multi-scale + ACRN + focal/class weight) on the small-disk AVA v2.2 subset.
# Model from the full-pipeline config, data/schedule from the subset config.

_base_ = ["./ava22_swin-tiny_fgtn_subset.py"]

custom_imports = dict(
    imports=["projects.fgtn_ava.fgtn_neck", "projects.fgtn_ava.heads"], allow_failed_imports=False
)

# e.g. "data/ava/annotations/class_weight_train_subset.json" from make_class_weight.py
class_weight = None

model = dict(
    backbone=dict(out_indices=(2, 3)),
    neck=dict(in_channels=(384, 768)),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_stride=16),
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

# stage-3 sequences are 4x more numerous than stage-4 ones: halve the batch to stay within VRAM
train_dataloader = dict(batch_size=4)
work_dir = "./work_dirs/ava22_swin-tiny_fgtn-ms-acrn-focal_subset"
