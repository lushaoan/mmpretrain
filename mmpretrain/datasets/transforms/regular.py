"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-10-09 11:11:21
Copyright: Copyright (c) 2022
LastEditTime: 2024-10-09 11:11:21
"""

from typing import Dict, List, Tuple

import cv2
from mmcv.transforms import BaseTransform, Compose

from mmpretrain.registry import TRANSFORMS


# 用于一维码统一方向，横轴是长边，纵轴是短边
@TRANSFORMS.register_module()
class Regularize(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: Dict) -> dict:
        img = results["img"]
        h = img.shape[0]
        w = img.shape[1]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        results["img"] = img

        return results
