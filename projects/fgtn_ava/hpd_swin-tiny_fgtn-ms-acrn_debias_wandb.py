# A8 on human_pose_dataset_v2: multi-scale Hybrid FGTN-Mamba (Swin stages 3+4) + ACRN + focal,
# on the debias recipe so it pairs with work_dirs/hpd_debias/{A0_noneck,A5_hybrid}.
# Model block = ava22_swin-tiny_fgtn-ms-acrn-focal_32x2x1-10e.py (class_weight=None: focal only).
_base_ = ["./hpd_swin-tiny_fgtn_debias_wandb.py"]

custom_imports = dict(
    imports=[
        "projects.fgtn_ava.fgtn_neck",
        "projects.fgtn_ava.hpd_dataset",
        "projects.fgtn_ava.debias_transforms",
        "projects.fgtn_ava.heads",
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(out_indices=(2, 3)),
    neck=dict(in_channels=(384, 768)),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_stride=16),
        shared_head=dict(type="ACRNHead", in_channels=2304, out_channels=1152),
        bbox_head=dict(type="BBoxHeadAVAWeighted", in_channels=1152, focal_gamma=2.0, focal_alpha=1.0),
    ),
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="WandbVisBackend", init_kwargs=dict(project="fgtn-hpd", group="debias")),
]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)
