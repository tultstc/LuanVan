"""Transforms that take scene / camera shortcuts away from the AVA-style action head.

In human_pose_dataset_v2 every dangerous_action clip comes from one 720x576 camera and
every other class from ~1080p cameras, so image size, sharpness, colour and background
all predict the label. These transforms remove (eval + train) or randomise (train) those
cues, and mask the context so the head must use the person.
"""
import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS

MEAN = np.array([123.675, 116.28, 103.53])  # the data preprocessor maps this colour to 0


def _pixel_boxes(results):
    h, w = results["img_shape"]
    boxes = np.asarray(results["proposals"], np.float32)[:, :4]
    if boxes.max() <= 1.0:  # still normalized (RawFrameDecode only rescales with gt_bboxes)
        boxes = boxes * np.array([w, h, w, h], np.float32)
    return boxes


@TRANSFORMS.register_module()
class FillBoxes(BaseTransform):
    """With probability p, fill the inside (people) or the outside (context) of all
    proposal boxes with MEAN."""

    def __init__(self, region, p=1.0):
        assert region in ("inside", "outside")
        self.region, self.p = region, p

    def transform(self, results):
        if np.random.rand() >= self.p:
            return results
        h, w = results["img_shape"]
        mask = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in np.round(_pixel_boxes(results)).astype(int):
            mask[max(y1, 0):y2, max(x1, 0):x2] = True
        if self.region == "outside":
            mask = ~mask
        for img in results["imgs"]:
            img[mask] = MEAN.astype(img.dtype)
        return results


@TRANSFORMS.register_module()
class RandomDegrade(BaseTransform):
    """Per clip: random down-up rescale (sharpness), JPEG re-encode (compression) and
    grayscale, so camera quality stops identifying the source."""

    def __init__(self, scale=(0.3, 1.0), jpeg_q=(30, 95), p_blur=0.5, p_jpeg=0.5, p_gray=0.2):
        self.scale, self.jpeg_q = scale, jpeg_q
        self.p_blur, self.p_jpeg, self.p_gray = p_blur, p_jpeg, p_gray

    def transform(self, results):
        s = np.random.uniform(*self.scale) if np.random.rand() < self.p_blur else 1.0
        q = int(np.random.randint(*self.jpeg_q)) if np.random.rand() < self.p_jpeg else None
        gray = np.random.rand() < self.p_gray
        out = []
        for img in results["imgs"]:
            h, w = img.shape[:2]
            if s < 1.0:
                small = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_AREA)
                img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            if q is not None:
                img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_UNCHANGED)
            if gray:
                img = np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
            out.append(img)
        results["imgs"] = out
        return results


if __name__ == "__main__":
    img = np.full((4, 6, 3), 255, np.uint8)
    r = FillBoxes("inside").transform(dict(img_shape=(4, 6), proposals=np.array([[0, 0, 0.5, 0.5]]), imgs=[img]))
    assert (r["imgs"][0][:2, :3] == MEAN.astype(np.uint8)).all() and (r["imgs"][0][2:, 3:] == 255).all()
    img = np.full((4, 6, 3), 255, np.uint8)
    r = FillBoxes("outside").transform(dict(img_shape=(4, 6), proposals=np.array([[0, 0, 3, 2]]), imgs=[img]))
    assert (r["imgs"][0][:2, :3] == 255).all() and (r["imgs"][0][2:] == MEAN.astype(np.uint8)).all()
    clip = [np.random.randint(0, 255, (32, 48, 3), np.uint8) for _ in range(3)]
    r = RandomDegrade(p_blur=1, p_jpeg=1, p_gray=1).transform(dict(imgs=clip))
    assert all(i.shape == (32, 48, 3) and i.dtype == np.uint8 for i in r["imgs"])
    assert (r["imgs"][0][..., 0] == r["imgs"][0][..., 1]).all()
    print("debias_transforms selfcheck ok")
