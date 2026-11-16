"""Re-split human_pose_dataset_v2 into train/val by source video (no shared scenes).

The shipped split is per person box: every video, and 75% of val keyframes, also appear
in train, and `segment_1_7_<n>_<a>-<b>` clips are overlapping 3 s windows of one recording.
This merges both splits, groups clips by source recording and assigns whole groups to val
(seed-shuffled greedy, ~20% of each class's boxes, capped at 30%). A class whose groups are
all too large (sleep_on_table: one video) stays train-only and has no val AP.

Usage (repo root, numpy env):
    python projects/fgtn_ava/make_hpd_split.py --root data/hpd/human_pose_dataset_v2
Writes annotations/{train,val}_vsplit.csv, proposals_{train,val}_vsplit.pkl and empty
*_vsplit_excluded_timestamps.csv next to the originals.
"""
import argparse
import csv
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def source_group(video_id):
    # segment_1_7_3354_3353.0-3356.0 -> segment_1_7 ; segment_100_99.0-102.0 -> segment
    return re.sub(r"_\d+_[\d.]+-[\d.]+$", "", video_id)


def split_groups(group_counts, seed, target=0.2, cap=0.3):
    total = Counter()
    for c in group_counts.values():
        total.update(c)
    order = sorted(group_counts)
    random.Random(seed).shuffle(order)
    val, val_cnt = set(), Counter()
    for g in order:
        c = group_counts[g]
        if all(val_cnt[k] < target * total[k] and val_cnt[k] + n <= cap * total[k] for k, n in c.items()):
            val.add(g)
            val_cnt.update(c)
    return val, val_cnt, total


def merge_proposals(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            if k in out and not np.array_equal(out[k], v):
                out[k] = np.unique(np.concatenate([out[k], v]), axis=0)  # union of boxes
            else:
                out.setdefault(k, v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hpd/human_pose_dataset_v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    anno = Path(args.root) / "annotations"

    rows = []
    for f in ("train.csv", "val.csv"):
        rows += [tuple(r) for r in csv.reader(open(anno / f)) if r]
    rows = sorted(set(rows), key=lambda r: (r[0], int(r[1]), r[7], r[6]))

    group_counts = defaultdict(Counter)
    for r in rows:
        group_counts[source_group(r[0])][r[6]] += 1
    val_groups, val_cnt, total = split_groups(group_counts, args.seed)

    split = {"train": [], "val": []}
    for r in rows:
        split["val" if source_group(r[0]) in val_groups else "train"].append(r)

    props = merge_proposals(*(pickle.load(open(anno / f, "rb")) for f in ("proposals_train.pkl", "proposals_val.pkl")))
    for name, rs in split.items():
        with open(anno / f"{name}_vsplit.csv", "w", newline="") as f:
            csv.writer(f).writerows(rs)
        keys = {f"{r[0]},{int(r[1]):04d}" for r in rs}
        with open(anno / f"proposals_{name}_vsplit.pkl", "wb") as f:
            pickle.dump({k: props[k] for k in keys if k in props}, f)
        (anno / f"{name}_vsplit_excluded_timestamps.csv").write_text("")
        missing = sum(k not in props for k in keys)
        print(f"{name}: {len(rs)} boxes, {len(keys)} keyframes, {len({r[0] for r in rs})} clips, "
              f"{len({source_group(r[0]) for r in rs})} sources, keyframes w/o proposals: {missing}")

    for k in sorted(total):
        print(f"class {k}: val {val_cnt[k]}/{total[k]} = {val_cnt[k] / total[k]:.1%}")
    tv = {source_group(r[0]) for r in split["train"]} & {source_group(r[0]) for r in split["val"]}
    assert not tv, f"source groups in both splits: {tv}"
    print("val sources:", sorted(val_groups))


if __name__ == "__main__":
    main()
