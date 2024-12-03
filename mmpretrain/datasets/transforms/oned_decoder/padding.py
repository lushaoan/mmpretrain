"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-12 18:39:57
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-12 18:39:58
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class OffsetAndPadIndices(BaseTransform):
    def __init__(self, offset: int, target_len: int, pad_value: int) -> None:
        self.offset = offset
        self.target_len = target_len
        self.pad_value = pad_value

    def transform(self, results: dict) -> dict:
        raw_indices = results["indices"]
        assert isinstance(raw_indices, np.ndarray)
        indices_len = len(results["indices"])
        padded_indices = np.zeros(self.target_len, raw_indices.dtype)

        padded_indices[:indices_len] = raw_indices
        padded_indices += self.offset

        padded_indices[indices_len:] = self.pad_value
        results["indices"] = torch.from_numpy(padded_indices)

        return results
