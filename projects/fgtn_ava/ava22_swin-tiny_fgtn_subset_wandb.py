# AVA v2.2 subset run with Weights & Biases logging (same model/data/schedule as the base).
# Needs `wandb login` (or WANDB_API_KEY) on the training machine; run name comes from
# WANDB_NAME (set per variant by run_ablation.sh).

_base_ = ["./ava22_swin-tiny_fgtn_subset.py"]

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="WandbVisBackend", init_kwargs=dict(project="fgtn-ava22-subset")),
]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)
