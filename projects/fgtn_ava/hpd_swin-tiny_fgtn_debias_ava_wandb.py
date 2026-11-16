# Debias hpd run + AVA people doing fight/phone/sleep in the train set (make_hpd_ava_extra.py),
# so fights no longer come from one camera only. Val/test stay hpd-only.
_base_ = ["./hpd_swin-tiny_fgtn_debias_wandb.py"]

hpd_anno = "data/hpd/human_pose_dataset_v2/annotations"
ava_train = dict(
    type="AVADataset",
    ann_file=f"{hpd_anno}/ava_extra_train.csv",
    exclude_file=None,  # applied when the csv was built
    proposal_file=f"{hpd_anno}/proposals_ava_extra_train.pkl",
    label_file=f"{hpd_anno}/label_map.pbtxt",
    data_prefix=dict(img="data/ava/rawframes"),
    num_classes=5,
    pipeline={{_base_.train_pipeline}},
)
train_dataloader = dict(
    dataset=dict(_delete_=True, type="ConcatDataset", datasets=[{{_base_.train_dataloader.dataset}}, ava_train])
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="WandbVisBackend", init_kwargs=dict(project="fgtn-hpd", group="debias_ava")),
]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)
