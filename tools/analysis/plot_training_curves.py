"""Publication-quality training curves from an mmengine AVA training log.

Parses the real values logged during training (no resampling / fabrication):
train loss every `logger.interval` iters, val mAP per epoch. Loss is shown
raw (light) + EMA-smoothed (solid) — standard presentation for noisy curves.

Usage:
    python tools/analysis/plot_training_curves.py trainning.log --out figures/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRAIN_RE = re.compile(r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\].*?\bloss: ([\d.]+)")
VAL_RE = re.compile(
    r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\]\s+mAP/overall: ([\d.]+)\s+"
    r"mAP/person_movement: ([\d.]+)\s+mAP/object_manipulation: ([\d.]+)\s+"
    r"mAP/person_interaction: ([\d.]+)"
)

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def parse_log(path):
    loss_x, loss_y, val = [], [], []
    for line in open(path):
        m = TRAIN_RE.search(line)
        if m:
            epoch, it, per_epoch, loss = int(m[1]), int(m[2]), int(m[3]), float(m[4])
            loss_x.append(epoch - 1 + it / per_epoch)  # fractional epoch
            loss_y.append(loss)
            continue
        m = VAL_RE.search(line)
        if m:
            val.append([int(m[1])] + [float(g) for g in m.groups()[1:]])
    return np.array(loss_x), np.array(loss_y), np.array(val)


def ema(y, alpha=0.05):
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="mmengine training log file")
    parser.add_argument("--out", default="figures", help="output directory")
    parser.add_argument("--title", default="Hybrid FGTN-Mamba (Swin-T) on AVA v2.2 subset")
    args = parser.parse_args()

    loss_x, loss_y, val = parse_log(args.log)
    if not len(loss_x):
        raise SystemExit(f"no train-loss lines found in {args.log}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0))

    # (a) training loss: raw + EMA
    ax1.plot(loss_x, loss_y, color="#9ecae1", lw=0.6, alpha=0.8, label="raw (per 20 iters)")
    ax1.plot(loss_x, ema(loss_y), color="#08519c", lw=1.6, label="EMA smoothed")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss (action cls, BCE)")
    ax1.set_xlim(0, loss_x.max())
    ax1.legend(loc="upper right")
    ax1.set_title("(a) Training loss", loc="left")

    # (b) validation mAP@0.5IoU per epoch
    if len(val):
        ep = val[:, 0]
        series = [
            ("Overall", val[:, 1], "#08519c", "o", 1.8),
            ("Person movement", val[:, 2], "#6baed6", "s", 1.2),
            ("Person interaction", val[:, 4], "#fd8d3c", "^", 1.2),
            ("Object manipulation", val[:, 3], "#74c476", "D", 1.2),
        ]
        for name, y, c, mk, lw in series:
            ax2.plot(ep, y * 100, color=c, marker=mk, ms=4, lw=lw, label=name)
        best = val[:, 1].max() * 100
        ax2.annotate(
            f"best {best:.2f}",
            xy=(ep[val[:, 1].argmax()], best),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#08519c",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation mAP@0.5 IoU (%)")
        ax2.set_xticks(ep.astype(int))
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize=8.5)
        ax2.set_title("(b) Validation mAP", loc="left")

    fig.suptitle(args.title, y=1.03, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"training_curves.{ext}")
    print(f"wrote {out}/training_curves.png + .pdf "
          f"({len(loss_y)} loss points, {len(val)} val epochs)")


if __name__ == "__main__":
    main()
