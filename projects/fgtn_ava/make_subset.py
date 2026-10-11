"""Build a small AVA v2.2 subset: filter annotations + proposals to N videos.

Picks the first N_train / N_val video ids (sorted, so deterministic) and writes
`*_subset.*` siblings next to the originals, plus a download list for
download_ava.sh. Disk budget scales linearly: ~1 GB rawframes per video at
short-side 256.

Usage (after the `annotations` and `proposals` stages):
    python projects/fgtn_ava/make_subset.py --n-train 20 --n-val 6
"""

import argparse
import csv
import pickle
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Filter AVA v2.2 annotations to a video subset")
    parser.add_argument("--anno-dir", default="data/ava/annotations")
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None, help="random video pick (default: first N sorted)")
    return parser.parse_args()


def unique_video_ids(csv_path):
    with open(csv_path) as f:
        return sorted({row[0] for row in csv.reader(f) if row})


def filter_csv(src, dst, keep):
    with open(src) as fin, open(dst, "w", newline="") as fout:
        writer = csv.writer(fout)
        n = 0
        for row in csv.reader(fin):
            if row and row[0] in keep:
                writer.writerow(row)
                n += 1
    print(f"  {dst.name}: {n} rows")


def filter_proposals(src, dst, keep):
    # keys are "<video_id>,<timestamp>"
    with open(src, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    sub = {k: v for k, v in data.items() if k.split(",")[0] in keep}
    with open(dst, "wb") as f:
        pickle.dump(sub, f)
    print(f"  {dst.name}: {len(sub)}/{len(data)} keys")


def main():
    args = parse_args()
    anno = Path(args.anno_dir)

    all_train = unique_video_ids(anno / "ava_train_v2.2.csv")
    all_val = unique_video_ids(anno / "ava_val_v2.2.csv")
    if args.seed is not None:
        rng = random.Random(args.seed)
        train_ids = sorted(rng.sample(all_train, args.n_train))
        val_ids = sorted(rng.sample(all_val, args.n_val))
    else:
        train_ids = all_train[: args.n_train]
        val_ids = all_val[: args.n_val]
    keep = set(train_ids) | set(val_ids)
    print(f"subset: {len(train_ids)} train + {len(val_ids)} val videos")

    for split, ids in (("train", set(train_ids)), ("val", set(val_ids))):
        filter_csv(anno / f"ava_{split}_v2.2.csv", anno / f"ava_{split}_v2.2_subset.csv", ids)
        filter_csv(
            anno / f"ava_{split}_excluded_timestamps_v2.2.csv",
            anno / f"ava_{split}_excluded_timestamps_v2.2_subset.csv",
            ids,
        )
        pkl = anno / f"ava_dense_proposals_{split}.FAIR.recall_93.9.pkl"
        if pkl.exists():
            filter_proposals(pkl, anno / f"ava_dense_proposals_{split}.FAIR.recall_93.9_subset.pkl", ids)
        else:
            print(f"  [skip] {pkl.name} not found — run the proposals stage first")

    # download list: resolve extensions from the full trainval file list
    list_file = anno / "ava_file_names_trainval_v2.1.txt"
    names = [line.strip() for line in open(list_file) if line.strip()]
    subset_names = [n for n in names if n.rsplit(".", 1)[0] in keep]
    out_list = anno / "ava_file_names_subset.txt"
    out_list.write_text("\n".join(subset_names) + "\n")
    print(f"  {out_list.name}: {len(subset_names)} videos to download")


if __name__ == "__main__":
    main()
