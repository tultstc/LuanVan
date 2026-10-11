_base_ = ["../e2e_thumos_fgtn_swin_tiny.py"]

# local windowed-MHSA branch only (W=5)
model = dict(projection=dict(branch_mode="attn_only"))

work_dir = "exps/thumos/e2e_fgtn_attn_only_swin_tiny"
