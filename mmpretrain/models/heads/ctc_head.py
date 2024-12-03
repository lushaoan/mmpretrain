"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 16:27:49
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 16:27:49
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from fast_ctc_decode import viterbi_search as viterbi_search_imp
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.models.backbones.multi_stage import MetaNeXtBlock
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


def _viterbi_search2(
    prob_txc: np.ndarray, collapse_repeats=True
) -> Tuple[str, List[int]]:
    assert isinstance(prob_txc, np.ndarray)
    assert prob_txc.ndim == 2

    BLANK = 0

    labels = []
    output_time_steps = []

    last_label = None
    for t, probs_1xc in enumerate(prob_txc):
        # find label with max prob in current time-step
        label = np.argmax(probs_1xc)

        if label != BLANK and (not collapse_repeats or last_label != label):
            labels.append(label)
            output_time_steps.append(t)

        last_label = label

    return labels, output_time_steps


def viterbi_search(network_output_txc, index_offset=0) -> tuple:
    if isinstance(network_output_txc, np.ndarray):
        network_output_txc = torch.from_numpy(network_output_txc)

    assert isinstance(network_output_txc, torch.Tensor)

    prob_txc = torch.softmax(network_output_txc, dim=1)
    assert isinstance(prob_txc, torch.Tensor)

    # convert to numpy
    prob_txc = prob_txc.detach().cpu().numpy()

    num_classes = prob_txc.shape[-1]
    alphabet = ["x"] + [f"{i + index_offset} " for i in range(num_classes - 1)]

    content, path1 = viterbi_search_imp(network_output=prob_txc, alphabet=alphabet)
    assert isinstance(content, str)

    content = content.strip()

    if not content:
        return tuple()  # empty tuple

    indices = tuple(int(i) for i in content.split(" "))

    indices2, path2 = _viterbi_search2(prob_txc)

    np.testing.assert_equal(indices2, indices)
    np.testing.assert_equal(path1, path2)

    return indices


@MODELS.register_module()
class CTCHead(BaseModule):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        local_attn_expand_ratio: int,
        blank: int,
        num_classes: int,
        loss: dict,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.local_attn = nn.Identity()
        if local_attn_expand_ratio > 0:
            self.local_attn = MetaNeXtBlock(
                dim=in_channels,
                expand_ratio=local_attn_expand_ratio,
                token_mixer_setting={
                    "type": "ConvNeXtTokenMixer",
                    "kernel_h": 1,
                    "kernel_w": 3,
                },
                norm_layer=nn.BatchNorm2d,
            )

        if mid_channels <= 0:
            self.pre_fc_conv = nn.Identity()
            self.fc = nn.Conv2d(
                in_channels,
                num_classes,
                kernel_size=1,
                bias=True,
            )
        else:
            self.pre_fc_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU6(),
            )
            self.fc = nn.Conv2d(
                mid_channels,
                num_classes,
                kernel_size=1,
                bias=True,
            )

        self.loss_module = MODELS.build(loss)
        self.index_offset = blank + 1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]):
        last_bchw = features[-1]
        last_bchw = self.local_attn(last_bchw)
        last_bchw = self.pre_fc_conv(last_bchw)

        predict_bc1w = self.fc(last_bchw)

        return predict_bc1w

    def _cvt_tensor_bchw2btc(self, tensor_bchw: torch.Tensor) -> torch.Tensor:
        bct = tensor_bchw.squeeze(dim=2)
        btc = torch.transpose(bct, 1, 2)

        return btc

    def loss(
        self, features: List[torch.Tensor], data_samples: List[DataSample]
    ) -> torch.Tensor:
        predict_bchw = self(features)
        predict_btc = self._cvt_tensor_bchw2btc(predict_bchw)
        targets_bxt = torch.stack([item.indices for item in data_samples])
        targets_lengths_bx1 = torch.stack([item.indices_len for item in data_samples])
        targets_lengths_bx1 = torch.reshape(targets_lengths_bx1, (-1, 1))

        losses = dict()
        loss = self.loss_module(predict_btc, targets_bxt, targets_lengths_bx1)
        losses["loss"] = loss

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None,
    ):
        predict_bc1w = self(feats)

        return predict_bc1w

    @classmethod
    def decode_result(cls, predicts_txc: torch.Tensor, index_offset: int):
        predicts_txc = predicts_txc.detach().cpu().numpy()
        predict_indices = viterbi_search(predicts_txc, index_offset)

        predict_indices = torch.as_tensor(predict_indices)

        return predict_indices
