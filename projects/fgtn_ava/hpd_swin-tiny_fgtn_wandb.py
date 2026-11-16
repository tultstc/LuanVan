# Hybrid FGTN-Mamba on human_pose_dataset_v2 (4 workshop behaviours), W&B logging.
# Prereq (repo root): python projects/fgtn_ava/make_hpd_split.py  (video-level re-split)
# Frames: rawframes_256 = rawframes resized to short side 256 (see README).
# Timestamps in this dataset are frame indices, so fps=1 and timestamp_start=0.

_base_ = ["./ava22_swin-tiny_fgtn_32x2x1-10e.py"]

custom_imports = dict(
    imports=["projects.fgtn_ava.fgtn_neck", "projects.fgtn_ava.hpd_dataset"], allow_failed_imports=False
)

num_classes = 5  # 4 behaviours + background
# topk feeds the logged recall/prec@topk only; the default (3, 5) needs > 5 classes
model = dict(roi_head=dict(bbox_head=dict(num_classes=num_classes, topk=(1, 3))))

dataset_type = "AVAVarLenDataset"
hpd_root = "data/hpd/human_pose_dataset_v2"
data_root = f"{hpd_root}/rawframes_256"
anno_root = f"{hpd_root}/annotations"
label_file = f"{anno_root}/label_map.pbtxt"
ann_file_train = f"{anno_root}/train_vsplit.csv"
ann_file_val = f"{anno_root}/val_vsplit.csv"
exclude_file_train = f"{anno_root}/train_vsplit_excluded_timestamps.csv"
exclude_file_val = f"{anno_root}/val_vsplit_excluded_timestamps.csv"
proposal_file_train = f"{anno_root}/proposals_train_vsplit.pkl"
proposal_file_val = f"{anno_root}/proposals_val_vsplit.pkl"

frame_args = dict(
    type=dataset_type,
    data_prefix=dict(img=data_root),
    label_file=label_file,
    num_classes=num_classes,
    filename_tmpl="frame_{:05}.jpg",
    start_index=0,
    fps=1,
    timestamp_start=0,
    timestamp_end=1820,  # only a fallback; real per-clip length comes from total_frames
)
train_dataloader = dict(
    dataset=dict(
        **frame_args,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        proposal_file=proposal_file_train,
    )
)
val_dataloader = dict(
    dataset=dict(
        **frame_args,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        proposal_file=proposal_file_val,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=ann_file_val, exclude_file=exclude_file_val, label_file=label_file, num_classes=num_classes
)
test_evaluator = val_evaluator

# single GPU: scale lr linearly vs the 8x6 recipe, as in the AVA subset runs
auto_scale_lr = dict(enable=True)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="WandbVisBackend", init_kwargs=dict(project="fgtn-hpd")),
]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)
