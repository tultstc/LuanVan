_base_ = ["../e2e_thumos_fgtn_swin_tiny.py"]

# both branches, fixed 0.5/0.5 fusion instead of the learned sigmoid gate
model = dict(projection=dict(branch_mode="no_gate"))

work_dir = "exps/thumos/e2e_fgtn_no_gate_swin_tiny"
