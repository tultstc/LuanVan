"""Class-weighted focal BCE head for AVA-style multi-label action classification.

mmaction2's `BBoxHeadAVA` already supports focal modulation (`focal_gamma`,
`focal_alpha`); this subclass adds a fixed per-class weight on top, giving the
two-level loss used in the thesis:

    L = mean_{i,c} [ w_c * alpha * (1 - p_t)^gamma * BCE(logit_ic, y_ic) ]

`weighted_focal_bce` holds the loss math on its own so it can be checked
without mmaction installed (see selfcheck.py); the head only wires it into the
mmaction2 RoI pipeline.
"""

import json

import torch
import torch.nn.functional as F


def weighted_focal_bce(cls_score, labels, gamma=0.0, alpha=1.0, class_weight=None):
    """Element-wise focal BCE with an optional per-class weight.

    cls_score, labels: (N, K) logits and {0,1} targets.
    class_weight: None or tensor/list of shape (K,).
    Returns the scalar mean loss; gamma=0, alpha=1, class_weight=None recovers
    plain `binary_cross_entropy_with_logits` exactly (mmaction2's default).
    """
    loss = F.binary_cross_entropy_with_logits(cls_score, labels, reduction="none")
    pt = torch.exp(-loss)
    loss = alpha * (1 - pt) ** gamma * loss
    if class_weight is not None:
        w = torch.as_tensor(class_weight, dtype=loss.dtype, device=loss.device)
        assert w.shape == (loss.shape[1],), f"class_weight has {w.numel()} entries, expected {loss.shape[1]}"
        loss = loss * w
    return loss.mean()


def _load_class_weight(class_weight):
    if class_weight is None or not isinstance(class_weight, str):
        return class_weight
    with open(class_weight) as f:  # json list written by make_class_weight.py
        return json.load(f)


try:
    from mmaction.models.roi_heads.bbox_heads.bbox_head import BBoxHeadAVA
    from mmaction.registry import MODELS

    @MODELS.register_module()
    class BBoxHeadAVAWeighted(BBoxHeadAVA):
        """`BBoxHeadAVA` + per-class loss weight.

        class_weight: list of length `num_classes - 1` when `background_class`
        is True (AVA label ids 1..80, background column dropped as in the parent),
        otherwise `num_classes`; or a path to a json list (make_class_weight.py).
        """

        def __init__(self, *args, class_weight=None, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.multilabel, "BBoxHeadAVAWeighted is written for the multi-label (sigmoid) setting"
            w = _load_class_weight(class_weight)
            if w is not None:
                expected = self.num_classes - 1 if self.background_class else self.num_classes
                assert len(w) == expected, f"class_weight has {len(w)} entries, expected {expected}"
                self.register_buffer("class_weight", torch.tensor(w, dtype=torch.float32))
            else:
                self.class_weight = None

        def loss_and_target(self, cls_score, rois, sampling_results, rcnn_train_cfg, **kwargs):
            # same bookkeeping as the parent; only the final loss expression differs
            cls_targets = self.get_targets(sampling_results, rcnn_train_cfg)
            labels, _ = cls_targets

            losses = dict()
            if cls_score is not None:
                if self.background_class:
                    labels = labels[:, 1:]
                    cls_score = cls_score[:, 1:]
                pos_inds = torch.sum(labels, dim=-1) > 0
                cls_score = cls_score[pos_inds]
                labels = labels[pos_inds]

                recall_thr, prec_thr, recall_k, prec_k = self.topk_accuracy(cls_score, labels, thr=0.5)
                losses["recall@thr=0.5"] = recall_thr
                losses["prec@thr=0.5"] = prec_thr
                for i, k in enumerate(self.topk):
                    losses[f"recall@top{k}"] = recall_k[i]
                    losses[f"prec@top{k}"] = prec_k[i]

                losses["loss_action_cls"] = weighted_focal_bce(
                    cls_score, labels, self.focal_gamma, self.focal_alpha, self.class_weight
                )

            return dict(loss_bbox=losses, bbox_targets=cls_targets)

except ImportError:
    pass  # standalone use (selfcheck.py) without mmaction installed
