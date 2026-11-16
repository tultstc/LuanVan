"""End-to-end per-class Precision / Recall / F1 (detector boxes + action model) on val keyframes
that have ground truth, plus the person detector's recall (thesis Table 5.2).

Usage (repo root): python projects/fgtn_ava/eval_e2e.py GT_vsplit.csv DUMP_ROOT A0_noneck A5_hybrid A8_full [--thr 0.5]
DUMP_ROOT/<model>/<video>.json are written by demo_video.py --dump.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from score_val_demos import iou, score_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("root")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    gt = defaultdict(lambda: defaultdict(dict))  # video -> frame -> box -> [box, labels]
    for v, ts, x1, y1, x2, y2, lab, _ in csv.reader(open(args.gt)):
        gt[v][int(ts)].setdefault((x1, y1, x2, y2), [[float(x1), float(y1), float(x2), float(y2)], set()])[1].add(int(lab) - 1)
    for m in args.models:
        c, n_gt, n_hit, n_kf, names = np.zeros((3, 4), int), 0, 0, 0, None
        for v in gt:
            d = json.load(open(Path(args.root) / m / f"{v}.json"))
            W, H = d["size"]
            names = d["labels"]
            for t, fr in d["frames"].items():
                if int(t) not in gt[v]:
                    continue
                g = [(np.multiply(b, [W, H, W, H]), ls) for b, ls in gt[v][int(t)].values()]
                c += np.array(score_frame(fr["boxes"], fr["scores"], g, args.thr))
                n_kf, n_gt = n_kf + 1, n_gt + len(g)
                if len(fr["boxes"]):
                    n_hit += int((iou(np.array(fr["boxes"], float), np.array([x[0] for x in g])) >= 0.5).any(0).sum())
        tp, fp, fn = c
        print(f"{m}: thr {args.thr}, {n_kf} keyframes, {n_gt} GT persons, detector recall (IoU>=0.5) {n_hit / n_gt:.3f}")
        has_gt = (tp + fn) > 0
        for k in np.flatnonzero(has_gt):
            print(f"  {names[k]:18s} P {tp[k] / max(tp[k] + fp[k], 1):.3f}  R {tp[k] / (tp[k] + fn[k]):.3f}  "
                  f"F1 {2 * tp[k] / (2 * tp[k] + fp[k] + fn[k]):.3f}  (tp {tp[k]} fp {fp[k]} fn {fn[k]})")
        T = c[:, has_gt].sum(1)
        print(f"  micro (classes with GT) P {T[0] / (T[0] + T[1]):.3f}  R {T[0] / (T[0] + T[2]):.3f}  F1 {2 * T[0] / (2 * T[0] + T[1] + T[2]):.3f}")
        for k in np.flatnonzero(~has_gt):
            print(f"  {names[k]:18s} no GT in val, false positives {fp[k]}")


if __name__ == "__main__":
    main()
