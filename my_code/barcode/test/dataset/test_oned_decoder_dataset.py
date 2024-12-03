"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-12 16:43:15
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-12 16:43:16
"""

import sys

sys.path.append("/dataset/shaoanlu/github/mmlab/mmpretrain")
import torch

from mmpretrain.datasets import OnedDecoderDataset

if __name__ == "__main__":
    dataset = OnedDecoderDataset(
        dataset_path=["/dataset/shaoanlu/dataset/tmp/gen128/"], pipeline=[]
    )
    print(dataset[0])
    print(type(dataset[0]))
