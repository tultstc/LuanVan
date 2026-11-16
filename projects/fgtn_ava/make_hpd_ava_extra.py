"""Extra hpd training data from the local AVA v2.2 videos (train + val annotations).

In hpd every dangerous_action (fight) clip comes from one camera that shows nothing else,
so the camera predicts the label. AVA movies show fights, phone use and sleeping in the
same videos, which breaks that link. AVA labels are mapped onto the hpd label ids; keyframes
with none of them are dropped, so only labelled people are added (as hpd: proposals = GT).

Usage (repo root):
    python projects/fgtn_ava/make_hpd_ava_extra.py
Writes <hpd>/annotations/ava_extra_train.csv and proposals_ava_extra_train.pkl.
"""
import argparse
import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# AVA id -> hpd id (1 use_mobilephone, 2 sleep_on_table, 4 dangerous_action)
AVA_TO_HPD = {15: 1, 57: 1, 8: 2, 64: 4, 71: 4, 76: 4}  # answer/text phone, lie/sleep, fight/kick/push


def convert(rows, excluded, have):
    out = set()
    for r in rows:
        lab = AVA_TO_HPD.get(int(r[6]))
        if lab and r[0] in have and (r[0], int(r[1])) not in excluded:
            out.add((r[0], int(r[1]), *r[2:6], lab, r[7]))
    return sorted(out, key=lambda r: (r[0], r[1], r[7], r[6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ava", default="data/ava")
    ap.add_argument("--hpd", default="data/hpd/human_pose_dataset_v2")
    args = ap.parse_args()
    ava, out_dir = Path(args.ava), Path(args.hpd) / "annotations"

    have = {p.name for p in (ava / "rawframes").iterdir() if p.is_dir()}
    rows, excluded = [], set()
    for s in ("train", "val"):
        rows += list(csv.reader(open(ava / f"annotations/ava_{s}_v2.2.csv")))
        excluded |= {(v, int(t)) for v, t in csv.reader(open(ava / f"annotations/ava_{s}_excluded_timestamps_v2.2.csv"))}
    out = convert(rows, excluded, have)

    props = defaultdict(set)
    for r in out:
        props[f"{r[0]},{r[1]:04d}"].add(tuple(map(float, r[2:6])))
    props = {k: np.array([[*b, 1.0] for b in sorted(v)], np.float32) for k, v in props.items()}

    # every keyframe's centre frame must exist (15-min cut from 900 s, 30 fps, img_00001 first)
    n_frames = {v: len(list((ava / "rawframes" / v).glob("img_*.jpg"))) for v in {r[0] for r in out}}
    assert all((r[1] - 900) * 30 + 1 <= n_frames[r[0]] for r in out)

    with open(out_dir / "ava_extra_train.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    with open(out_dir / "proposals_ava_extra_train.pkl", "wb") as f:
        pickle.dump(props, f)
    boxes = Counter(r[6] for r in out)
    vids = Counter(lab for lab, _ in {(r[6], r[0]) for r in out})
    print(f"{len(out)} rows, {len(props)} keyframes, {len(n_frames)}/{len(have)} videos")
    for k in sorted(boxes):
        print(f"hpd class {k}: {boxes[k]} boxes in {vids[k]} videos")


def _selfcheck():
    rows = [["v", "0902", "0", "0", "1", "1", "64", "3"], ["v", "0902", "0", "0", "1", "1", "76", "3"],
            ["v", "0903", "0", "0", "1", "1", "15", "1"], ["v", "0904", "0", "0", "1", "1", "80", "1"],
            ["w", "0902", "0", "0", "1", "1", "8", "1"], ["v", "0905", "0", "0", "1", "1", "8", "1"]]
    out = convert(rows, excluded={("v", 905)}, have={"v"})
    assert out == [("v", 902, "0", "0", "1", "1", 4, "3"), ("v", 903, "0", "0", "1", "1", 1, "1")], out


if __name__ == "__main__":
    _selfcheck()
    main()
