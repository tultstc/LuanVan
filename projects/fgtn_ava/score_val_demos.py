"""Score demo_video.py --dump files against the val ground truth, per video and model.

Per keyframe with GT: detections matched to GT boxes greedily (IoU >= 0.5); on matched people
labels with score >= thr count TP/FP vs the GT label set; unmatched GT labels are FN, labels on
unmatched detections FP. Prints F1 per video for each model.
usage: python score_val_demos.py GT.csv DUMP_ROOT MODEL... [--thr 0.5]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], 1)
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def score_frame(det, scores, gt, thr, K=4):
    """det (D,4), scores (D,K), gt list of (box(4,), set of 0-based labels), thr scalar or (K,)
    -> per-class tp, fp, fn arrays (K,)"""
    tp, fp, fn = np.zeros(K, int), np.zeros(K, int), np.zeros(K, int)
    y = [np.isin(np.arange(K), list(g[1])) for g in gt]
    p = [np.asarray(s) >= thr for s in scores]
    used_d, used_g = set(), set()
    if len(det) and gt:
        m = iou(np.asarray(det, float), np.array([g[0] for g in gt], float))
        for d, g in sorted(np.argwhere(m >= 0.5).tolist(), key=lambda q: -m[q[0], q[1]]):
            if d in used_d or g in used_g:
                continue
            used_d.add(d), used_g.add(g)
            tp, fp, fn = tp + (p[d] & y[g]), fp + (p[d] & ~y[g]), fn + (~p[d] & y[g])
    for g in set(range(len(gt))) - used_g:
        fn = fn + y[g]
    for d in set(range(len(det))) - used_d:
        fp = fp + p[d]
    return tp, fp, fn


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("root")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5], help="one value, or one per class")
    ap.add_argument("--fit", action="store_true", help="pick the F1-best threshold per class and model")
    args = ap.parse_args()
    args.thr = np.array(args.thr)  # broadcasts against each detection's (K,) scores

    gt = defaultdict(lambda: defaultdict(dict))  # vid -> ts -> pid -> [box, labels]
    for v, ts, x1, y1, x2, y2, lab, pid in csv.reader(open(args.gt)):
        e = gt[v][int(ts)].setdefault(pid, [[float(x1), float(y1), float(x2), float(y2)], set()])
        e[1].add(int(lab) - 1)

    # model -> video -> list of (boxes, scores, gt) on keyframes that have GT
    frames = defaultdict(dict)
    for m in args.models:
        for v in sorted(gt):
            p = Path(args.root) / m / f"{v}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            W, H = d["size"]
            frames[m][v] = [(fr["boxes"], fr["scores"], [(np.multiply(b, [W, H, W, H]), ls) for b, ls in gt[v][int(t)].values()])
                            for t, fr in d["frames"].items() if int(t) in gt[v]]

    def counts(m, vids, thr):
        c = np.zeros((3, 4), int)
        for v in vids:
            for det, sc, g in frames[m][v]:
                c += np.array(score_frame(det, sc, g, thr))
        return c

    thr = {m: args.thr for m in args.models}
    if args.fit:  # per-class threshold maximising that class's F1 over all val videos (thesis C4)
        grid = np.round(np.arange(0.02, 0.99, 0.02), 2)
        for m in args.models:
            c = np.stack([counts(m, frames[m], t) for t in grid])  # (len(grid), 3, K)
            f = f1(c[:, 0], c[:, 1], c[:, 2])
            has_gt = (c[0, 0] + c[0, 2]) > 0  # tp + fn: a class absent from val keeps the default
            thr[m] = np.where(has_gt, grid[f.argmax(0)], np.broadcast_to(args.thr, has_gt.shape))
            print(f"{m}: thr {thr[m].tolist()}  class F1 {f.max(0).round(2).tolist()}  (at 0.5: {f[grid.tolist().index(0.5)].round(2).tolist()})")

    print(f"{'video':28s}{'class':>8s}{'kf':>5s}" + "".join(f"{m[:10]:>12s}" for m in args.models))
    for v in sorted(gt):
        classes = sorted({l for t in gt[v].values() for _, ls in t.values() for l in ls})
        row = [float(f1(*counts(m, [v], thr[m]).sum(1))) if v in frames[m] else float("nan") for m in args.models]
        n_kf = max((len(frames[m].get(v, [])) for m in args.models), default=0)
        print(f"{v:28s}{'+'.join(str(c + 1) for c in classes):>8s}{n_kf:>5d}" + "".join(f"{x:12.2f}" for x in row))


if __name__ == "__main__":
    gt1 = [(np.array([0, 0, 10, 10.0]), {0}), (np.array([20, 20, 30, 30.0]), {3})]
    # person 1 right, person 2 missed, one extra det predicting class 1 -> tp1 fp1 fn1
    tp, fp, fn = score_frame([[0, 0, 10, 10], [50, 50, 60, 60]], [[0.9, 0, 0, 0], [0, 0.8, 0, 0]], gt1, 0.5)
    assert (tp.tolist(), fp.tolist(), fn.tolist()) == ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1])
    tp, fp, fn = score_frame([[0, 0, 10, 10]], [[0.3, 0, 0, 0]], gt1, np.array([0.2, 0.5, 0.5, 0.5]))
    assert (tp.sum(), fp.sum(), fn.sum()) == (1, 0, 1)  # per-class threshold catches the 0.3 score
    assert sum(x.sum() for x in score_frame([], [], gt1, 0.5)) == 2
    main()
