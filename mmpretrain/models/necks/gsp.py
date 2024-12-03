"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-12 17:44:15
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-12 17:44:15
"""

"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-11 15:34:22
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-11 15:34:22
"""

"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 11:30:25
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 11:30:26
"""
from typing import List

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class GlobalStripPooling(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_axis: str
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        if pool_axis == "w":
            kernel_size = (1, kernel_size)
        else:
            kernel_size = (kernel_size, 1)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=1,
            bias=False,
            groups=in_channels,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    m.weight.data.fill_(1 / self.kernel_size)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, features) -> List[torch.Tensor]:
        last_feature_map = features[-1]
        out = self.conv(last_feature_map)

        return features[:-1] + [out]
