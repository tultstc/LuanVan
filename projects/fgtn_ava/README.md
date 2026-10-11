# Hybrid FGTN-Mamba on AVA v2.2 (Video Swin-Tiny)

Spatio-temporal action detection on AVA v2.2 with MMAction2. The FGTN neck
(`fgtn_neck.py`) sits between the Video Swin-Tiny backbone and the AVA RoI head,
applying hybrid Mamba + windowed-MHSA temporal reasoning along T at every
spatial position of the `(B, 768, 16, H/32, W/32)` feature map.

## Setup (GPU machine)

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0,<2.2.0"   # full mmcv: RoIAlign ops required
mim install mmdet                            # FastRCNN / RoI head
mim install "mmaction2>=1.2.0"
# bidirectional Mamba (bimamba v2) — Video-Mamba-Suite fork, CUDA required:
pip install causal-conv1d
pip install git+https://github.com/OpenGVLab/video-mamba-suite.git#subdirectory=mamba
```

Without `mamba-ssm` the neck falls back to a pure-PyTorch scan (correct but
~100x slower) — fine for smoke tests, not for training.

## Data: AVA v2.2

One script, staged and resumable (run from repo root):

```bash
bash projects/fgtn_ava/download_ava.sh                # everything, in order
# or stage by stage:
bash projects/fgtn_ava/download_ava.sh annotations    # v2.2 csv/pbtxt (~15 MB)
bash projects/fgtn_ava/download_ava.sh proposals      # FAIR dense proposals (~700 MB)
NUM_JOBS=8 bash projects/fgtn_ava/download_ava.sh videos   # 299 videos (~160 GB)
bash projects/fgtn_ava/download_ava.sh cut frames     # ffmpeg: trim 15-30min + 30fps jpg (~500 GB)
```

Reserve ~700 GB of disk for the full pipeline. `DATA_DIR=<path>` overrides the
default `data/ava`. URLs mirror mmaction2's official
[tools/data/ava](https://github.com/open-mmlab/mmaction2/tree/main/tools/data/ava) scripts.

### Small-disk subset (~25-30 GB)

Filter everything to N videos, extract frames at short-side 256, and delete
intermediates as soon as each video is processed:

```bash
bash projects/fgtn_ava/download_ava.sh annotations proposals
python projects/fgtn_ava/make_subset.py --n-train 45 --n-val 8 --seed 42
NUM_JOBS=4 bash projects/fgtn_ava/download_ava.sh videos          # subset list auto-detected
CLEAN=1 JPEG_Q=5 bash projects/fgtn_ava/download_ava.sh cut frames
rm -rf data/ava/videos_15min                                       # frames are all that training needs
```

`--seed` picks videos randomly (better class coverage than the first-N default);
`JPEG_Q=5` keeps 53 videos within ~30 GB (~0.55 GB/video, fine for 256px input).

Then train with the subset config (same neck/ablation options):

```bash
mim train mmaction2 projects/fgtn_ava/ava22_swin-tiny_fgtn_subset.py --gpus 1
```

Budget guide: ~1 GB rawframes per video at short-side 256 / q:v 3 — tune
`--n-train/--n-val` to your disk. Peak usage during `cut frames` with `CLEAN=1`
stays around (frames so far) + ~1 GB per in-flight video. Subset numbers are
not comparable to published AVA results, but all neck variants trained on the
same subset remain a valid controlled comparison for the ablation study.

Expected layout (matches the config):

```
data/ava/
├── annotations/
│   ├── ava_train_v2.2.csv / ava_val_v2.2.csv
│   ├── ava_*_excluded_timestamps_v2.2.csv
│   ├── ava_action_list_v2.2_for_activitynet_2019.pbtxt
│   └── ava_dense_proposals_{train,val}.FAIR.recall_93.9.pkl
└── rawframes/
```

## Train

Run **from this repo's root** (custom_imports resolves `projects.fgtn_ava.fgtn_neck`):

```bash
mim train mmaction2 projects/fgtn_ava/ava22_swin-tiny_fgtn_32x2x1-10e.py \
    --gpus 8 --launcher pytorch
```

## Ablations

Same config, one override each (`branch_mode`):

```bash
--cfg-options model.neck.branch_mode=mamba_only   # S3TN: SSM branch only
--cfg-options model.neck.branch_mode=attn_only    # local windowed MHSA only
--cfg-options model.neck.branch_mode=no_gate      # fixed 0.5/0.5 fusion
```

Baseline "TAN gốc" (no neck): `--cfg-options model.neck=None` — or compare
against mmaction2's published SlowFast/VideoMAE AVA 2.2 numbers.

## Test

```bash
mim test mmaction2 projects/fgtn_ava/ava22_swin-tiny_fgtn_32x2x1-10e.py \
    --checkpoint work_dirs/.../best_mAP_epoch_*.pth --gpus 8 --launcher pytorch
```

The `AVAMetric` evaluator reports per-class AP natively (all 60 evaluated AVA
classes) — use it for the slow-vs-fast per-class analysis.

## Local self-check (no GPU needed)

```bash
python -c "
import torch; from projects.fgtn_ava.fgtn_neck import FGTNNeckAVA
n = FGTNNeckAVA(768, num_blocks=2); x = torch.randn(1, 768, 16, 7, 7)
assert n(x).shape == x.shape; print('ok')"
```
