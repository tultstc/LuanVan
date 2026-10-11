# Small-disk AVA v2.2 subset run (~25-30 GB of rawframes).
# Prereq: python projects/fgtn_ava/make_subset.py  (writes the *_subset files)

_base_ = ["./ava22_swin-tiny_fgtn_32x2x1-10e.py"]

anno_root = "data/ava/annotations"

ann_file_train = f"{anno_root}/ava_train_v2.2_subset.csv"
ann_file_val = f"{anno_root}/ava_val_v2.2_subset.csv"
exclude_file_train = f"{anno_root}/ava_train_excluded_timestamps_v2.2_subset.csv"
exclude_file_val = f"{anno_root}/ava_val_excluded_timestamps_v2.2_subset.csv"
proposal_file_train = f"{anno_root}/ava_dense_proposals_train.FAIR.recall_93.9_subset.pkl"
proposal_file_val = f"{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9_subset.pkl"

train_dataloader = dict(
    batch_size=8,  # ~37 GB on a 48 GB card (28 GB @ 6 in the first run)
    dataset=dict(
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        proposal_file=proposal_file_train,
    )
)
val_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        proposal_file=proposal_file_val,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=ann_file_val, exclude_file=exclude_file_val)
test_evaluator = val_evaluator

# small-data schedule: first run was still improving at epoch 10 -> train 2x longer
train_cfg = dict(max_epochs=20)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=True, begin=0, end=2, convert_to_iter_based=True),
    dict(type="CosineAnnealingLR", T_max=18, eta_min=0, by_epoch=True, begin=2, end=20, convert_to_iter_based=True),
]

# single GPU: scale lr linearly vs the 8x6 recipe (1.25e-4 * 8/48)
auto_scale_lr = dict(enable=True)
