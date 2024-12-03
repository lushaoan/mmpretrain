"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 14:06:03
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 14:06:03
"""

import sys

import torch

sys.path.append("/dataset/shaoanlu/github/mmlab/mmpretrain")
from mmpretrain.models.necks import GlobalStripPooling

if __name__ == "__main__":
    neck = GlobalStripPooling(160, 160, 8, "h")
    fake_input = torch.rand(1, 160, 8, 64)
    print(neck)
    out = neck([fake_input])
