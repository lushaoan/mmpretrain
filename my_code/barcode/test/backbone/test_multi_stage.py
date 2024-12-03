"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 15:31:38
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 15:31:39
"""

import sys

sys.path.append("/dataset/shaoanlu/github/mmlab/mmpretrain")
import torch

from mmpretrain.models.backbones import MultiStageBackbone

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

    backbone = MultiStageBackbone(in_channels=3, arch_settings=arch_settings)
    print(backbone)
    backbone.init_weights()
    imgs = torch.randn(1, 3, 64, 512)
    feats = backbone(imgs)
    for ele in feats:
        print(ele.shape)
