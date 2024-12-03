"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 16:07:30
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 16:07:30
"""

from typing import List, Optional

import torch
import torch.nn as nn
from mmengine.structures import BaseDataElement

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier


@MODELS.register_module()
class OnedDecoder(BaseClassifier):
    def __init__(
        self,
        backbone: dict,
        neck: dict,
        head: dict,
        pretrained: Optional[str] = None,
        train_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super(OnedDecoder, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor
        )

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.head = MODELS.build(head)

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        mode: str = "tensor",
    ) -> dict:
        if mode == "tensor":
            feats = self.extract_feat(inputs)
            return self.head(feats)
        elif mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        x = self.backbone(inputs)
        x = self.neck(x)

        return x

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample]) -> dict:
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def predict(
        self, inputs: torch.Tensor, data_samples: Optional[List[DataSample]] = None
    ) -> List[DataSample]:
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples)
