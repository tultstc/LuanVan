"""Per-class loss weights from an AVA annotation csv (class-balanced, Cui et al. 2019).

    w_c = (1 - beta) / (1 - beta^{n_c}),  then rescaled so that mean_c w_c = 1

n_c = number of training boxes carrying label c (a box with several labels counts
once per label). Classes never seen in the csv get n_c = 1 (largest weight).
Output: json list of length `num_classes` (AVA: 80, label ids 1..80 in order),
to be passed as `bbox_head.class_weight` in the config.

Usage (from repo root):
    python projects/fgtn_ava/make_class_weight.py \
        --ann data/ava/annotations/ava_train_v2.2_subset.csv \
        --out data/ava/annotations/class_weight_train_subset.json
"""

import argparse
import csv
import json
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Class-balanced loss weights from an AVA csv")
    p.add_argument("--ann", required=True, help="AVA train csv (video,sec,x1,y1,x2,y2,label,entity)")
    p.add_argument("--out", required=True, help="json path for the weight list")
    p.add_argument("--num-classes", type=int, default=80, help="AVA label ids 1..num_classes")
    p.add_argument("--beta", type=float, default=0.999, help="effective-number hyper-parameter")
    p.add_argument("--clip", type=float, default=10.0, help="upper bound on w_c after rescaling")
    return p.parse_args()


def class_weights(counts, num_classes, beta, clip):
    n = [max(counts.get(c, 0), 1) for c in range(1, num_classes + 1)]
    w = [(1 - beta) / (1 - beta**k) for k in n]
    mean = sum(w) / len(w)
    return [min(x / mean, clip) for x in w]


def main():
    args = parse_args()
    counts = Counter()
    with open(args.ann) as f:
        for row in csv.reader(f):
            if row:
                counts[int(row[6])] += 1
    w = class_weights(counts, args.num_classes, args.beta, args.clip)
    with open(args.out, "w") as f:
        json.dump([round(x, 4) for x in w], f)
    seen = sum(1 for c in range(1, args.num_classes + 1) if counts.get(c, 0) > 0)
    print(f"{sum(counts.values())} boxes, {seen}/{args.num_classes} classes seen; "
          f"w in [{min(w):.3f}, {max(w):.3f}] -> {args.out}")


if __name__ == "__main__":
    main()
