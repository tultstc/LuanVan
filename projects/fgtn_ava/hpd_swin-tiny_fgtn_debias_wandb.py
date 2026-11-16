# hpd run with the scene/camera shortcuts taken away (see debias_transforms.py):
#  - every frame resized to the same 384x256 (3:2), train and eval: frame size/aspect no
#    longer tells 720x576 (dangerous_action camera) from 1080p. No crop, so no box is lost.
#  - train only: colour jitter + random blur/JPEG/grayscale (camera quality),
#    and the context outside the person boxes masked in half the clips (forces the person).
_base_ = ["./hpd_swin-tiny_fgtn_wandb.py"]

custom_imports = dict(
    imports=[
        "projects.fgtn_ava.fgtn_neck",
        "projects.fgtn_ava.hpd_dataset",
        "projects.fgtn_ava.debias_transforms",
    ],
    allow_failed_imports=False,
)

file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="SampleAVAFrames", clip_len=32, frame_interval=2),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="Resize", scale=(384, 256), keep_ratio=False),
    dict(type="RandomRescale", scale_range=(256, 320)),
    dict(type="RandomCrop", size=256),
    dict(type="ColorJitter", brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    dict(type="RandomDegrade"),
    dict(type="FillBoxes", region="outside", p=0.5),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="SampleAVAFrames", clip_len=32, frame_interval=2, test_mode=True),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="Resize", scale=(384, 256), keep_ratio=False),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    dict(type="PackActionInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="WandbVisBackend", init_kwargs=dict(project="fgtn-hpd", group="debias")),
]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)
