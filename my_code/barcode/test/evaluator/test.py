"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-14 14:28:27
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-14 14:28:28
"""

import mmengine.dist as dist
from torchvision.datasets import CelebA

if dist.get_rank() == 0:
    data = ["foo", {1: 2}]
else:
    data = [24, {"a": "b"}]
output = dist.collect_results_cpu(data, 4)
print(output)
