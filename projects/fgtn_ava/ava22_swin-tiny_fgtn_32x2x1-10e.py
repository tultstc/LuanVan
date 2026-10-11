# Hybrid FGTN-Mamba neck + Video Swin-Tiny on AVA v2.2.
# Based on mmaction2's slowfast ava22 recipe; backbone/neck/optimizer adapted.
# Run from the repo root (custom_imports resolves projects.fgtn_ava.fgtn_neck).

custom_imports = dict(imports=["projects.fgtn_ava.fgtn_neck"], allow_failed_imports=False)

# Kinetics-400 pretrained Video Swin-Tiny (recognizer checkpoint, backbone.* keys)
url = (
    "https://download.openmmlab.com/mmaction/v1.0/recognition/swin/"
    "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/"
    "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth"
)

model = dict(
    type="FastRCNN",
    _scope_="mmdet",
    init_cfg=dict(type="Pretrained", checkpoint=url),
    backbone=dict(
        type="mmaction.SwinTransformer3D",
        arch="tiny",
        pretrained=None,
        pretrained2d=False,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
    ),
    # FGTN: temporal reasoning on the (B, 768, 16, H/32, W/32) map before RoI pooling
    neck=dict(
        type="mmaction.FGTNNeckAVA",
        in_channels=768,
        num_blocks=2,
        branch_mode="hybrid",  # hybrid | mamba_only | attn_only | no_gate
        n_head=4,
        window_size=5,
    ),
    roi_head=dict(
        type="AVARoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor3D",
            roi_layer_type="RoIAlign",
            output_size=8,
            with_temporal_pool=True,
        ),
        bbox_head=dict(
            type="BBoxHeadAVA",
            background_class=True,
            in_channels=768,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5,
        ),
    ),
    data_preprocessor=dict(
        type="mmaction.ActionDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape="NCTHW",
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssignerAVA",
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9,
            ),
            sampler=dict(
                type="RandomSampler",
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=1.0,
        )
    ),
    test_cfg=dict(rcnn=None),
)

dataset_type = "AVADataset"
data_root = "data/ava/rawframes"
anno_root = "data/ava/annotations"

ann_file_train = f"{anno_root}/ava_train_v2.2.csv"
ann_file_val = f"{anno_root}/ava_val_v2.2.csv"

exclude_file_train = f"{anno_root}/ava_train_excluded_timestamps_v2.2.csv"
exclude_file_val = f"{anno_root}/ava_val_excluded_timestamps_v2.2.csv"

label_file = f"{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt"

proposal_file_train = f"{anno_root}/ava_dense_proposals_train.FAIR.recall_93.9.pkl"
proposal_file_val = f"{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl"

file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="SampleAVAFrames", clip_len=32, frame_interval=2),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="RandomRescale", scale_range=(256, 320)),
    dict(type="RandomCrop", size=256),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    dict(type="PackActionInputs"),
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type="SampleAVAFrames", clip_len=32, frame_interval=2, test_mode=True),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=6,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="AVAMetric",
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val,
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=True, begin=0, end=2, convert_to_iter_based=True),
    dict(type="CosineAnnealingLR", T_max=8, eta_min=0, by_epoch=True, begin=2, end=10, convert_to_iter_based=True),
]

# AdamW for the transformer backbone (mirrors mmaction2's ViT AVA recipe)
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1.25e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=40, norm_type=2),
)

auto_scale_lr = dict(enable=False, base_batch_size=48)

# runtime (inlined from mmaction2's _base_/default_runtime.py)
default_scope = "mmaction"

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20, ignore_last=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="auto", max_keep_ckpts=2),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)

log_level = "INFO"
load_from = None
resume = False
