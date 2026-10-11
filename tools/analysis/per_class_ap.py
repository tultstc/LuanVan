"""Per-class AP analysis for comparing temporal-neck variants (TAN / S3TN / FGTN).

Reuses the standard OpenTAD mAP evaluator, but reports the per-class AP matrix
it normally averages away, alongside each class's mean GT duration — so slow
vs fast action classes can be compared across neck variants.

Usage (single run):
    python tools/analysis/per_class_ap.py data/thumos-14/annotations/thumos_14_anno.json \
        --pred fgtn=exps/thumos/e2e_fgtn_hybrid_swin_tiny/result_detection.json

Usage (compare variants, first one is the reference for deltas):
    python tools/analysis/per_class_ap.py <gt.json> \
        --pred tan=<...>/result_detection.json s3tn=<...>/result_detection.json fgtn=<...>/result_detection.json \
        --subset test --tiou 0.3 0.4 0.5 0.6 0.7 --out per_class_ap.csv

Note: predictions require `post_processing.save_dict=True` when running tools/test.py.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opentad.evaluations import mAP  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Per-class AP comparison across neck variants")
    parser.add_argument("gt", help="ground truth annotation json")
    parser.add_argument(
        "--pred",
        nargs="+",
        required=True,
        help="one or more `name=path/to/result_detection.json` pairs; first is the delta reference",
    )
    parser.add_argument("--subset", default="test", help="annotation subset to evaluate (default: test)")
    parser.add_argument(
        "--tiou", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7], help="tIoU thresholds (THUMOS default)"
    )
    parser.add_argument("--out", default=None, help="optional output csv path")
    return parser.parse_args()


def evaluate_variant(name, pred_path, args):
    evaluator = mAP(
        ground_truth_filename=args.gt,
        prediction_filename=pred_path,
        subset=args.subset,
        tiou_thresholds=args.tiou,
    )
    evaluator.evaluate()
    # ap: (n_tiou, n_class) -> average over tIoU thresholds per class
    per_class = pd.Series(evaluator.ap.mean(axis=0), name=name)
    return evaluator, per_class


def main():
    args = parse_args()

    variants = []
    for pair in args.pred:
        name, _, path = pair.partition("=")
        assert path, f"--pred expects name=path, got: {pair}"
        variants.append((name, path))

    columns, evaluator = {}, None
    for name, path in variants:
        evaluator, per_class = evaluate_variant(name, path, args)
        columns[name] = per_class
        print(f"[{name}] average mAP @ {args.tiou}: {evaluator.average_mAP * 100:.2f}%")

    # class metadata from the last evaluator's ground truth (identical across variants)
    gt = evaluator.ground_truth
    gt = gt.assign(duration=gt["t-end"] - gt["t-start"])
    idx_to_class = {v: k for k, v in evaluator.activity_index.items()}

    table = pd.DataFrame(
        {
            "class": [idx_to_class[i] for i in range(len(idx_to_class))],
            "num_gt": gt.groupby("label").size().reindex(range(len(idx_to_class))).values,
            "avg_dur_s": gt.groupby("label")["duration"].mean().reindex(range(len(idx_to_class))).values,
        }
    )
    for name, per_class in columns.items():
        table[f"AP_{name}"] = per_class.values * 100

    ref = variants[0][0]
    for name, _ in variants[1:]:
        table[f"delta_{name}"] = table[f"AP_{name}"] - table[f"AP_{ref}"]

    # slow actions first: the spec's hypothesis is SSM helps long/slow, attention helps short/fast
    table = table.sort_values("avg_dur_s", ascending=False).reset_index(drop=True)

    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))

    if args.out:
        table.to_csv(args.out, index=False, float_format="%.4f")
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
