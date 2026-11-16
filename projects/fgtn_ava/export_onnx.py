"""Export to ONNX for demo_video.py: the action model (backbone + neck + RoI head) or a
COCO person detector. Both are checked against PyTorch with onnxruntime after export.

Action model: inputs imgs (1, 3, 32, H, W) normalized RGB, rois (N, 5) [0, x1, y1, x2, y2] in
pixels of the H x W input; output scores (N, K) sigmoid, background column dropped.
Mamba's CUDA scan cannot be traced, so every Mamba module is switched to its python path with
mamba-ssm's own reference scan (selective_scan_ref); the check compares against the CUDA path.

Usage (repo root, GPU):
    PYTHONPATH=. python projects/fgtn_ava/export_onnx.py action CONFIG CKPT OUT.onnx
    PYTHONPATH=. python projects/fgtn_ava/export_onnx.py det OUT.onnx
"""
import argparse
import json
import re

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class ActionNet(nn.Module):
    """FastRCNN test path (mmaction AVARoIHead._bbox_forward) with tensors in and out."""

    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, imgs, rois):
        x = self.det.extract_feat(imgs)
        rh = self.det.roi_head
        feats, global_feat = rh.bbox_roi_extractor(x, rois)
        if rh.with_shared_head:
            feats = rh.shared_head(feats, feat=global_feat, rois=rois)
        bh = rh.bbox_head
        # = bh.forward in eval mode; its AdaptivePool3d((1, None, None)) has no ONNX export
        x = feats.mean(2, keepdim=True) if bh.temporal_pool_type == "avg" else feats.amax(2, keepdim=True)
        x = x.mean((3, 4), keepdim=True) if bh.spatial_pool_type == "avg" else x.amax((3, 4), keepdim=True)
        return bh.fc_cls(x.flatten(1)).sigmoid()[:, 1:]


def check(path, inputs, ref, tol):
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    outs = sess.run(None, {i.name: v.cpu().numpy() for i, v in zip(sess.get_inputs(), inputs)})
    ref = [r.detach().cpu().numpy() for r in ref]
    for o, r, t in zip(outs, ref, tol):
        assert o.shape == r.shape, (o.shape, r.shape)
        err = float(np.abs(o - r).max()) if o.size else 0.0
        print(f"onnxruntime vs torch: shape {o.shape}, max abs diff {err:.2e}")
        assert err <= t, f"ONNX output differs from PyTorch by {err} (tol {t})"


def labels_from_pbtxt(path):
    txt = open(path).read()
    return {int(i): n for n, i in re.findall(r'name:\s*"([^"]+)"\s*(?:label_)?id:\s*(\d+)', txt)}


def build_action(config, checkpoint):
    """ActionNet on CUDA in eval mode + its config (also used by demo_video.py)."""
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings

    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules

    cfg = Config.fromfile(config)
    register_all_modules()
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.init_cfg = None  # do not download Kinetics weights, the checkpoint has everything
    det = MODELS.build(cfg.model)
    load_checkpoint(det, checkpoint, map_location="cpu")
    return ActionNet(det).cuda().eval(), cfg


def export_action(args):
    net, cfg = build_action(args.config, args.checkpoint)

    w, h = args.size
    torch.manual_seed(0)
    imgs = torch.randn(1, 3, args.clip_len, h, w, device="cuda")
    rois = torch.tensor([[0, 40, 30, 200, 250], [0, 150, 10, 380, 240]], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        ref = net(imgs, rois)  # CUDA mamba kernel

    import mamba_ssm.modules.mamba_simple as ms
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    ms.selective_scan_fn = selective_scan_ref
    n = 0
    for m in net.modules():
        if isinstance(m, ms.Mamba):
            m.use_fast_path, n = False, n + 1
    print(f"{n} Mamba modules -> python path + selective_scan_ref")
    # mmcv RoIAlign's symbolic emits opset<13 Squeeze; torchvision's is the same op with an opset-16 export
    from torchvision.ops import roi_align

    rl = net.det.roi_head.bbox_roi_extractor.roi_layer
    assert rl.pool_mode == "avg", rl.pool_mode
    rl.forward = lambda f, r: roi_align(
        f, r, rl.output_size, rl.spatial_scale, rl.sampling_ratio, rl.aligned
    )

    with torch.no_grad():
        torch.onnx.export(
            net, (imgs, rois), args.out, opset_version=17,
            input_names=["imgs", "rois"], output_names=["scores"],
            dynamic_axes={"rois": {0: "n"}, "scores": {0: "n"}},
        )
    labels = labels_from_pbtxt(cfg.label_file)
    pp = cfg.model.data_preprocessor
    meta = dict(
        labels=json.dumps([labels[i] for i in sorted(labels)]),
        clip_len=str(args.clip_len), frame_interval=str(args.frame_interval),
        width=str(w), height=str(h), mean=json.dumps(list(pp.mean)), std=json.dumps(list(pp.std)),
    )
    model = onnx.load(args.out)
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, args.out)
    check(args.out, (imgs, rois), [ref], tol=[1e-3])


def export_det(args):
    import torchvision

    net = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").cuda().eval()
    img = torch.rand(3, 480, 640, device="cuda")
    with torch.no_grad():
        ref = net([img])[0]
        torch.onnx.export(
            net, ([img],), args.out, opset_version=17,
            input_names=["image"], output_names=["boxes", "labels", "scores"],
            dynamic_axes={"image": {1: "h", 2: "w"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
        )
    # boxes in pixels, labels exact, scores in [0, 1]
    check(args.out, (img,), [ref["boxes"], ref["labels"], ref["scores"]], tol=[0.5, 0, 1e-3])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    a = sub.add_parser("action")
    a.add_argument("config")
    a.add_argument("checkpoint")
    a.add_argument("out")
    a.add_argument("--size", type=int, nargs=2, default=(384, 256), metavar=("W", "H"))  # debias val Resize
    a.add_argument("--clip-len", type=int, default=32)
    a.add_argument("--frame-interval", type=int, default=2)
    d = sub.add_parser("det")
    d.add_argument("out")
    args = ap.parse_args()
    export_action(args) if args.what == "action" else export_det(args)


if __name__ == "__main__":
    main()
