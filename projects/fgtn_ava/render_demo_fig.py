"""Thesis Fig. 5.2 (fig:demo): same keyframe through A0/A6/A8 at a fixed 0.5 threshold, from the val dumps.
Per row the keyframe with the most correct minus wrong labels summed over the 3 models (fight row:
only keyframes where every model has exactly one box per GT person)."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from demo_video import COLORS
SHORT = ["phone", "sleep", "foot", "fight"]
def draw(img, boxes, scores, title, thr=0.5):
    h, w = img.shape[:2]; fs = w / 900; th = max(2, round(w / 400))
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = b.astype(int)
        hits = [(k, s[k]) for k in np.argsort(-s) if s[k] >= thr]
        col = COLORS[hits[0][0]] if hits else (180, 180, 180)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
        for j, (k, p) in enumerate(hits):
            txt = f"{SHORT[k]} {p:.2f}"; (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            y = max(y1 - 6 - j * (tht + 10), tht + 6)
            cv2.rectangle(img, (x1, y - tht - 6), (x1 + tw + 6, y + 6), (0, 0, 0), -1)
            cv2.putText(img, txt, (x1 + 3, y), cv2.FONT_HERSHEY_SIMPLEX, fs, COLORS[k], th)
    cv2.putText(img, title, (int(12 * fs), int(48 * fs)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * fs, (255, 255, 255), th + 1)
    return img
from score_val_demos import score_frame
GT, DUMP, IN, OUT = sys.argv[1:5]
MODELS = [("A0", "A0_noneck"), ("A6", "A5_hybrid"), ("A8", "A8_full")]
gt = defaultdict(lambda: defaultdict(dict))
for v, ts, x1, y1, x2, y2, lab, pid in csv.reader(open(GT)):
    gt[v][int(ts)].setdefault((x1, y1, x2, y2), [[float(x1), float(y1), float(x2), float(y2)], set()])[1].add(int(lab) - 1)
for row, vids in [("phone", ["segment_100_99.0-102.0", "segment_101_100.0-103.0", "segment_102_101.0-104.0", "segment_103_102.0-105.0", "segment_104_103.0-106.0"]),
                  ("foot", ["foot_on_table_3", "foot_on_table_1", "foot_on_table_2"]), ("fight", ["newfi62", "newfi42", "newfi46", "newfi49"])]:
    best = None
    for v in vids:
        d = {m: json.load(open(f"{DUMP}/{r}/{v}.json")) for m, r in MODELS}
        W, H = d["A0"]["size"]
        for t in d["A0"]["frames"]:
            if int(t) not in gt[v]: continue
            g = [(np.multiply(b, [W, H, W, H]), ls) for b, ls in gt[v][int(t)].values()]
            if row == "fight" and any(len(d[m]["frames"][t]["boxes"]) != len(g) for m, _ in MODELS): continue  # one box per fighter
            s = 0
            for m, _ in MODELS:
                fr = d[m]["frames"][t]; tp, fp, fn = score_frame(fr["boxes"], fr["scores"], g, 0.5)
                s += tp.sum() - fp.sum()
            if best is None or s > best[0]: best = (s, v, int(t), d)
    s, v, t, d = best
    cap = cv2.VideoCapture(f"{IN}/{v}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, frame = cap.read(); assert ok
    names = d["A0"]["labels"]
    for m, _ in MODELS:
        fr = d[m]["frames"][str(t)]
        img = draw(frame.copy(), np.array(fr["boxes"]).reshape(-1, 4), np.array(fr["scores"]).reshape(-1, 4), m)
        cv2.imwrite(f"{OUT}/demo_{m}_{row}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(row, v, "frame", t, "score", s)
