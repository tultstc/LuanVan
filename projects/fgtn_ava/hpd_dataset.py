"""AVA-format dataset for clips of different lengths (human_pose_dataset_v2)."""
import os

from mmaction.datasets import AVADataset
from mmaction.registry import DATASETS


@DATASETS.register_module()
class AVAVarLenDataset(AVADataset):
    """AVADataset gives every video the same shot (timestamp_start..timestamp_end), so a
    clip window near the end of a short video reads frames that do not exist. Passing each
    video's own frame count as `total_frames` makes SampleAVAFrames clip to it instead.
    Assumes contiguous frames numbered from `start_index` (checked for this dataset)."""

    def get_data_info(self, idx):
        info = super().get_data_info(idx)
        cache = self.__dict__.setdefault("_num_frames", {})
        d = info["frame_dir"]
        if d not in cache:
            cache[d] = sum(f.endswith(".jpg") for f in os.listdir(d))
        info["total_frames"] = cache[d]
        return info
