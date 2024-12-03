"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 16:14:11
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 16:14:12
"""

# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .oned_decoder import OnedDecoder
from .timm import TimmClassifier

__all__ = [
    "BaseClassifier",
    "ImageClassifier",
    "TimmClassifier",
    "HuggingFaceClassifier",
    "OnedDecoder",
]
