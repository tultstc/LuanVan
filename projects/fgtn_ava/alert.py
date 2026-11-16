"""Track-level alert logic (thesis Algorithm 3) and its evaluation on demo_video.py dumps.

Per keyframe: person boxes are linked to tracks by greedy IoU; for every (track, class) the last
n threshold decisions are kept and an event fires when at least m of them are positive and the
last event of the same (track, class) is at least `cooldown` seconds old.

Evaluation (val ground truth is annotated on every frame of each labelled range): an event is
correct if a GT person at that frame overlaps its box (IoU >= 0.5) and has its class; events on
frames without annotation are not scored. Per (video, class) present in GT we report whether a
correct event fired and the delay from the first GT frame of the class to the first correct event.

Usage (repo root):
    python projects/fgtn_ava/alert.py GT_vsplit.csv DUMP_DIR [--m 2 --n 3 --cooldown 5 --thr 0.5]
DUMP_DIR holds <video>.json written by demo_video.py --dump (keyframes every --stride frames).
"""
import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], 1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


class AlertLogic:
    def __init__(self, thr, m, n, cooldown, iou_thr=0.3, max_age=2):
        assert 1 <= m <= n
        self.thr, self.m, self.n, self.cd, self.iou_thr, self.max_age = np.asarray(thr), m, n, cooldown, iou_thr, max_age
        self.tracks = {}  # id -> dict(box, age)
        self.hist = defaultdict(lambda: deque(maxlen=n))  # (track, class) -> last n decisions
        self.last = defaultdict(lambda: -np.inf)  # (track, class) -> time of last event
        self.next_id = 0

    def _assign(self, boxes):
        ids = list(self.tracks)
        out = [None] * len(boxes)
        if ids and len(boxes):
            m = iou(boxes, [self.tracks[i]["box"] for i in ids])
            for d, t in sorted(np.argwhere(m >= self.iou_thr).tolist(), key=lambda q: -m[q[0], q[1]]):
                if out[d] is None and ids[t] not in out:
                    out[d] = ids[t]
        for d in range(len(boxes)):
            if out[d] is None:
                out[d], self.next_id = self.next_id, self.next_id + 1
        for i in list(self.tracks):  # tracks not seen for more than max_age keyframes are dropped
            self.tracks[i]["age"] += 1
            if self.tracks[i]["age"] > self.max_age:
                del self.tracks[i]
        for d, i in enumerate(out):
            self.tracks[i] = dict(box=boxes[d], age=0)
        return out

    def step(self, t, boxes, scores):
        """t seconds; boxes (P,4); scores (P,K) -> events [(t, track, box, class, score)]"""
        events = []
        for a, box, p in zip(self._assign(list(boxes)), boxes, scores):
            for c, s in enumerate(p):
                h = self.hist[a, c]
                h.append(s >= self.thr[c] if self.thr.ndim else s >= self.thr)
                if sum(h) >= self.m and t - self.last[a, c] >= self.cd:
                    events.append((t, a, list(box), c, float(s)))
                    self.last[a, c] = t
        return events


def evaluate(gt_csv, dump_dir, thr, m, n, cooldown, fps=25.0):
    gt = defaultdict(lambda: defaultdict(list))  # video -> frame -> [(box_norm, class0)]
    for v, ts, x1, y1, x2, y2, lab, _ in csv.reader(open(gt_csv)):
        gt[v][int(ts)].append(([float(x1), float(y1), float(x2), float(y2)], int(lab) - 1))
    rows, tot = [], dict(events=0, correct=0, wrong=0, unscored=0)
    for v in sorted(gt):
        p = Path(dump_dir) / f"{v}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        W, H = d["size"]
        logic, events = AlertLogic(thr, m, n, cooldown), []
        for f in sorted(d["frames"], key=int):
            fr = d["frames"][f]
            events += [(int(f),) + e for e in logic.step(int(f) / fps, fr["boxes"], fr["scores"])]
        first_ok = {}
        for f, t, a, box, c, s in events:
            tot["events"] += 1
            if f not in gt[v]:
                tot["unscored"] += 1
                continue
            g = [(np.multiply(b, [W, H, W, H]), k) for b, k in gt[v][f]]
            ok = any(k == c and iou(box, gb)[0, 0] >= 0.5 for gb, k in g)
            tot["correct" if ok else "wrong"] += 1
            if ok:
                first_ok.setdefault(c, f)
        for c in sorted({k for fr in gt[v].values() for _, k in fr}):
            start = min(f for f, fr in gt[v].items() if any(k == c for _, k in fr))
            delay = (first_ok[c] - start) / fps if c in first_ok else None
            rows.append((v, d["labels"][c], len(gt[v]), delay))
    return rows, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5])
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=5.0, help="seconds")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    thr = np.array(args.thr) if len(args.thr) > 1 else args.thr[0]
    for dump in args.dumps:
        rows, tot = evaluate(args.gt, dump, thr, args.m, args.n, args.cooldown, args.fps)
        det = [r for r in rows if r[3] is not None]
        delays = np.array([r[3] for r in det])
        print(f"\n{dump}: m/n={args.m}/{args.n} cooldown={args.cooldown}s thr={args.thr}")
        for cls in sorted({r[1] for r in rows}):
            rc = [r for r in rows if r[1] == cls]
            dc = np.array([r[3] for r in rc if r[3] is not None])
            print(f"  {cls:18s} episodes {len(rc):2d}  alerted {len(dc):2d}  "
                  f"delay median {np.median(dc) if len(dc) else float('nan'):.2f}s max {dc.max() if len(dc) else float('nan'):.2f}s")
        sc = tot["correct"] + tot["wrong"]
        print(f"  events {tot['events']}  scored {sc}  correct {tot['correct']}  wrong {tot['wrong']}  "
              f"precision {tot['correct'] / max(sc, 1):.3f}  unscored {tot['unscored']}  "
              f"episodes alerted {len(det)}/{len(rows)}  delay median {np.median(delays) if len(det) else float('nan'):.2f}s")


if __name__ == "__main__":
    # one person, class 0 above threshold on keyframes 0,1,2 -> m=2 of n=3 fires at the 2nd keyframe,
    # cooldown 1 s blocks the 3rd (0.32 s later); a second far-away person gets its own track
    lg = AlertLogic(0.5, 2, 3, 1.0)
    box, far = [0, 0, 10, 10], [50, 50, 60, 60]
    assert lg.step(0.0, [box], [[0.9, 0.1]]) == []
    ev = lg.step(0.32, [[1, 0, 11, 10], far], [[0.8, 0.1], [0.9, 0.0]])
    assert [(e[1], e[3]) for e in ev] == [(0, 0)], ev
    assert lg.step(0.64, [box, far], [[0.9, 0.1], [0.9, 0.0]]) == [(0.64, 1, far, 0, 0.9)]
    assert lg.step(1.40, [box], [[0.9, 0.1]])[0][1] == 0  # cooldown over for track 0
    main()
