"""Scene-shortcut probe: re-evaluate one checkpoint with the person, the context or the
motion removed from every val clip, and compare AP with the untouched input.

If AP barely drops when the people are masked (C1) but collapses when only the people are
kept (C2), the model classifies the scene rather than the behaviour. C3 (keyframe repeated)
shows how much the temporal signal matters.

Usage (repo root, GPU free):
    PYTHONPATH=. python projects/fgtn_ava/probe_scene.py CONFIG CHECKPOINT [--conditions ...]
"""
import argparse
import io
import json
import re
import sys

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.config import Config
from mmengine.runner import Runner

from mmaction.registry import TRANSFORMS
from projects.fgtn_ava.debias_transforms import MEAN, FillBoxes


@TRANSFORMS.register_module()
class StaticClip(BaseTransform):
    """Use the keyframe for every frame of the clip (no motion)."""

    def transform(self, results):
        inds = results["frame_inds"]
        results["frame_inds"] = np.full_like(inds, inds[len(inds) // 2])
        return results


CONDITIONS = {
    "C0_original": None,
    "C1_person_masked": ("RawFrameDecode", dict(type="FillBoxes", region="inside")),
    "C2_context_masked": ("RawFrameDecode", dict(type="FillBoxes", region="outside")),
    "C3_static": ("SampleAVAFrames", dict(type="StaticClip")),
}


class Tee(io.TextIOBase):
    """Keep a copy of stdout: ava_eval prints per-class AP instead of returning it."""

    def __init__(self, stream):
        self.stream, self.buf = stream, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()


def _selfcheck():
    img = np.full((4, 6, 3), 255, np.uint8)
    r = FillBoxes("inside").transform(dict(img_shape=(4, 6), proposals=np.array([[0, 0, 0.5, 0.5]]), imgs=[img]))
    assert (r["imgs"][0][:2, :3] == MEAN.astype(np.uint8)).all() and (r["imgs"][0][2:, 3:] == 255).all()
    r = StaticClip().transform(dict(frame_inds=np.arange(0, 64, 2)))
    assert (r["frame_inds"] == 32).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--work-dir", default="work_dirs/probe")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    args = ap.parse_args()
    _selfcheck()

    summary = {}
    for name in args.conditions:
        cfg = Config.fromfile(args.config)
        pipeline = cfg.test_dataloader.dataset.pipeline
        if CONDITIONS[name]:
            after, t = CONDITIONS[name]
            pipeline.insert(next(i for i, p in enumerate(pipeline) if p["type"] == after) + 1, t)
        cfg.visualizer.vis_backends = [dict(type="LocalVisBackend")]
        cfg.work_dir = f"{args.work_dir}/{name}"
        cfg.load_from = args.checkpoint
        tee = Tee(sys.stdout)
        sys.stdout = tee
        try:
            metrics = Runner.from_cfg(cfg).test()
        finally:
            sys.stdout = tee.stream
        per_class = dict(re.findall(r"Index: \d+, Action: (.*?): AP: ([\d.na]+);", tee.buf.getvalue()))
        summary[name] = dict(overall=float(metrics["mAP/overall"]), **{k: float(v) for k, v in per_class.items()})

    print("PROBE_SUMMARY " + json.dumps(summary))
    cols = list(next(iter(summary.values())))
    print(f"{'condition':20s}" + "".join(f"{c[:16]:>18s}" for c in cols))
    for name, row in summary.items():
        print(f"{name:20s}" + "".join(f"{100 * row[c]:18.2f}" for c in cols))


if __name__ == "__main__":
    main()
