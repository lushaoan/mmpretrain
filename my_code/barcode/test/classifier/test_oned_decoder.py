"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 16:15:32
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 16:15:32
"""

import sys

sys.path.append("/dataset/shaoanlu/github/mmlab/mmpretrain")
import torch

from mmpretrain.models.classifiers import OnedDecoder
from mmpretrain.registry import MODELS

if __name__ == "__main__":
    expand_ratio = 4.0
    token_mixer = {
        "type": "InceptionConcatDWConvTokenMixer",
        "res_scale": True,
        "square_kernel_size": 3,
        "dw_conv_w": 11,
        "dw_conv_h": 7,
    }

    arch_settings = [
        # num_blocks, channels, expand_ratio, stride_hw, token_mixer
        [1, 16, expand_ratio, (2, 2), token_mixer],
        [2, 24, expand_ratio, (2, 2), token_mixer],
        [3, 32, expand_ratio, (2, 2), token_mixer],
        [4, 64, expand_ratio, (1, 1), token_mixer],
        [3, 96, expand_ratio, (1, 1), token_mixer],
        [3, 160, expand_ratio, (1, 1), token_mixer],
    ]

    backbone_setting = {
        "type": "MultiStageBackbone",
        "in_channels": 3,
        "arch_settings": arch_settings,
    }
    neck_setting = {
        "type": "GlobalStripPooling",
        "in_channels": 160,
        "out_channels": 160,
        "kernel_size": 8,
        "pool_axis": "h",
    }
    head_setting = {
        "type": "CTCHead",
        "in_channels": 160,
        "mid_channels": 512,
        "local_attn_expand_ratio": 3,
        "num_classes": 235,
    }

    # decoder = OnedDecoder(
    #     backbone=backbone_setting, neck=neck_setting, head=head_setting
    # )
    # print(decoder)

    classifier_setting = {
        "type": "OnedDecoder",
        "backbone": backbone_setting,
        "neck": neck_setting,
        "head": head_setting,
    }
    decoder = MODELS.build(classifier_setting)
    print(decoder)
    imgs = torch.randn(1, 3, 64, 512)
    out = decoder(imgs)
    print(out.shape)
