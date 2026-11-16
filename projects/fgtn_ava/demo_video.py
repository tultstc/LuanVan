"""Run the action model on a video and write an annotated copy, with ONNX or PyTorch.

Backends, picked by the arguments:
  - ONNX (export_onnx.py): ACTION.onnx + DET.onnx; needs only onnxruntime, opencv, numpy.
  - PyTorch on GPU: ACTION.pth with --config; DET = ultralytics YOLO weights (*.pt) or
    "torchvision" (Faster R-CNN v2, COCO);
    the action model runs with mamba-ssm's CUDA kernel, as in training/evaluation.
Persons are detected on every `--stride`-th frame (keyframe). Each keyframe's clip is
`clip_len` frames, `frame_interval` apart, centred on it (as SampleAVAFrames at test time),
resized to the model input; labels are held until the next keyframe.

Usage (repo root):
    python projects/fgtn_ava/demo_video.py IN.mp4 ACTION.onnx DET.onnx OUT.mp4 [--title A5]
    PYTHONPATH=. python projects/fgtn_ava/demo_video.py IN.mp4 CKPT.pth yolo11x.pt OUT.mp4 --config CFG.py
"""
import argparse
import json

import cv2
import numpy as np

COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (0, 0, 255), (200, 0, 200)]  # BGR per class


def session(path):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def load_action(path, config):
    """-> run(clip (1,3,T,h,w), rois (N,5)) -> scores (N,K), and meta (labels, clip, input size, norm)"""
    if path.endswith(".onnx"):
        sess = session(path)
        m = sess.get_modelmeta().custom_metadata_map
        meta = dict(labels=json.loads(m["labels"]), clip_len=int(m["clip_len"]), interval=int(m["frame_interval"]),
                    size=(int(m["width"]), int(m["height"])), mean=json.loads(m["mean"]), std=json.loads(m["std"]))
        return (lambda clip, rois: sess.run(None, {"imgs": clip, "rois": rois})[0]), meta

    import torch

    from projects.fgtn_ava.export_onnx import build_action, labels_from_pbtxt

    assert config, "--config is required with a .pth checkpoint"
    net, cfg = build_action(config, path)
    pipe = {t["type"]: t for t in cfg.test_dataloader.dataset.pipeline}
    assert pipe["Resize"].get("keep_ratio") is False, "demo assumes a fixed-size Resize (debias val pipeline)"
    labels = labels_from_pbtxt(cfg.label_file)
    pp = cfg.model.data_preprocessor
    meta = dict(labels=[labels[i] for i in sorted(labels)], clip_len=pipe["SampleAVAFrames"]["clip_len"],
                interval=pipe["SampleAVAFrames"]["frame_interval"], size=tuple(pipe["Resize"]["scale"]),
                mean=list(pp.mean), std=list(pp.std))

    @torch.no_grad()
    def run(clip, rois):
        return net(torch.from_numpy(clip).cuda(), torch.from_numpy(rois).cuda()).cpu().numpy()

    return run, meta


def load_det(path):
    """-> detect(BGR frame, thr) -> person boxes (P, 4) in pixels"""
    if path.endswith(".pt"):  # ultralytics YOLO weights, COCO class 0 = person
        from ultralytics import YOLO

        yolo = YOLO(path)
        return lambda frame, thr: yolo.predict(frame, classes=[0], conf=thr, device=0, verbose=False)[0].boxes.xyxy.cpu().numpy()

    if path.endswith(".onnx"):
        sess = session(path)
        run = lambda img: sess.run(None, {"image": img})
    else:
        import torch
        import torchvision

        assert path == "torchvision", f"detector must be YOLO .pt, .onnx or 'torchvision', got {path}"
        net = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").cuda().eval()

        @torch.no_grad()
        def run(img):
            o = net([torch.from_numpy(img).cuda()])[0]
            return o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()

    def detect(frame, thr):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        boxes, labels, scores = run(img)
        return boxes[(labels == 1) & (scores >= thr)]  # torchvision COCO 1 = person

    return detect


def clip_indices(center, n_frames, clip_len, interval):
    start = center - (clip_len // 2) * interval
    return np.clip(start + np.arange(clip_len) * interval, 0, n_frames - 1)


def draw(frame, boxes, scores, names, thr, title, top1=False):
    for box, s in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        hits = [(names[k], s[k], k) for k in np.argsort(-s / thr) if s[k] >= thr[k]][: 1 if top1 else None]
        color = COLORS[hits[0][2] % len(COLORS)] if hits else (180, 180, 180)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for j, (name, p, k) in enumerate(hits):
            y = max(y1 - 8 - 22 * j, 18)
            cv2.putText(frame, f"{name} {p:.2f}", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[k % len(COLORS)], 2)
    if title:
        cv2.putText(frame, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("action", help="ACTION.onnx, or a .pth checkpoint with --config")
    ap.add_argument("det", help="YOLO weights (*.pt), DET.onnx, or 'torchvision'")
    ap.add_argument("out")
    ap.add_argument("--config", help="model config for a .pth checkpoint")
    ap.add_argument("--stride", type=int, default=8, help="run the action model every N frames")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5],
                    help="action score threshold for drawing: one value, or one per class")
    ap.add_argument("--top1", action="store_true", help="draw only the label with the highest score/threshold per person")
    ap.add_argument("--det-thr", type=float, default=0.7)
    ap.add_argument("--title", default="")
    ap.add_argument("--dump", help="also write per-keyframe boxes (pixels) and scores to this json")
    args = ap.parse_args()

    act, meta = load_action(args.action, args.config)
    det = load_det(args.det)
    names, clip_len, interval, (w, h) = meta["labels"], meta["clip_len"], meta["interval"], meta["size"]
    mean, std = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)
    assert len(args.thr) in (1, len(names)), f"--thr needs 1 or {len(names)} values"
    thr = np.broadcast_to(np.array(args.thr), (len(names),))

    # pass 1: model-size RGB frames for the clips + person boxes on keyframes
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    small, persons = [], {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i = len(small)
        H0, W0 = frame.shape[:2]
        if i % args.stride == 0:
            persons[i] = det(frame, args.det_thr)
        small.append(cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(small)
    assert n, "video has no frames"
    sx, sy = w / W0, h / H0

    results = {}
    for t, boxes in persons.items():
        if not len(boxes):
            results[t] = (boxes, np.zeros((0, len(names)), np.float32))
            continue
        clip = np.stack([small[j] for j in clip_indices(t, n, clip_len, interval)]).astype(np.float32)
        clip = ((clip - mean) / std).transpose(3, 0, 1, 2)[None]  # (1, 3, T, h, w)
        rois = np.concatenate([np.zeros((len(boxes), 1)), boxes * [sx, sy, sx, sy]], 1).astype(np.float32)
        scores = act(np.ascontiguousarray(clip), rois)
        results[t] = (boxes, scores)
        print(f"frame {t}/{n}: {len(boxes)} person(s), max score per class {scores.max(0).round(2).tolist()}")

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(dict(labels=names, size=[W0, H0], frames={
                t: dict(boxes=b.tolist(), scores=s.tolist()) for t, (b, s) in results.items()}), f)

    # pass 2: draw each frame with its latest keyframe result
    cap = cv2.VideoCapture(args.video)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W0, H0))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores = results[i - i % args.stride]
        out.write(draw(frame, boxes, scores, names, thr, args.title, args.top1))
    cap.release()
    out.release()
    print(f"wrote {args.out}: {n} frames, {len(persons)} keyframes")


if __name__ == "__main__":
    assert (clip_indices(0, 100, 4, 2) == [0, 0, 0, 2]).all() and (clip_indices(98, 100, 4, 2) == [94, 96, 98, 99]).all()
    main()
