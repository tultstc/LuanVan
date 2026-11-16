#!/usr/bin/env bash
# Sequential neck ablation (thesis Table 4.2: A0-A3, A5) on one GPU, same seed/hyperparameters.
# Resumable: re-run the script; finished variants (DONE marker) are skipped, an interrupted
# one resumes from its latest checkpoint.
#
# Usage (repo root):  bash projects/fgtn_ava/run_ablation.sh
# W&B: CFG=projects/fgtn_ava/ava22_swin-tiny_fgtn_subset_wandb.py (run name = variant id)
# Env: CFG=<config>  BS=6  SEED=42  VARIANTS="A5_hybrid A0_noneck ..."  WORK_ROOT=work_dirs/subset
#      EXTRA="key=val ..."  (extra --cfg-options appended to every run)

set -uo pipefail

CFG=${CFG:-projects/fgtn_ava/ava22_swin-tiny_fgtn_subset.py}
BS=${BS:-6}  # 28 GB/GPU in the logged run; 8 needs ~37 GB
SEED=${SEED:-42}
WORK_ROOT=${WORK_ROOT:-work_dirs/subset}
VARIANTS=${VARIANTS:-"A5_hybrid A0_noneck A1_mamba_only A2_attn_only A3_no_gate"}
EXTRA=${EXTRA:-}

# mim runs mmaction's own train.py, so the repo root must be importable for custom_imports
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

status=0
for v in ${VARIANTS}; do
    case "${v}" in
        A0_noneck) opt="model.neck=None" ;;
        A1_mamba_only) opt="model.neck.branch_mode=mamba_only" ;;
        A2_attn_only) opt="model.neck.branch_mode=attn_only" ;;
        A3_no_gate) opt="model.neck.branch_mode=no_gate" ;;
        A5_hybrid) opt="model.neck.branch_mode=hybrid" ;;
        *) echo "unknown variant ${v}"; exit 1 ;;
    esac
    wd="${WORK_ROOT}/${v}"
    if [[ -f "${wd}/DONE" ]]; then echo "==> ${v}: done, skip"; continue; fi
    echo "==> ${v}: ${opt}  -> ${wd}"
    # shellcheck disable=SC2086
    if WANDB_NAME="${v}" mim train mmaction2 "${CFG}" --gpus 1 --work-dir "${wd}" --resume \
        --cfg-options randomness.seed="${SEED}" train_dataloader.batch_size="${BS}" ${opt} ${EXTRA}; then
        touch "${wd}/DONE"
    else
        echo "==> ${v}: FAILED (re-run to resume)"; status=1
    fi
done
exit ${status}
