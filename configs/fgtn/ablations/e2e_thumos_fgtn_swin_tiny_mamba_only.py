_base_ = ["../e2e_thumos_fgtn_swin_tiny.py"]

# S3TN baseline: bidirectional selective SSM branch only
model = dict(projection=dict(branch_mode="mamba_only"))

work_dir = "exps/thumos/e2e_fgtn_mamba_only_swin_tiny"
