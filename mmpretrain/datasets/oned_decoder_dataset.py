"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-12 16:13:38
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-12 16:13:39
"""

import json
import os
from typing import List, Sequence

import numpy as np

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class OnedDecoderDataset(BaseDataset):
    def __init__(
        self,
        dataset_path: List[str],
        pipeline: Sequence = ...,
    ):
        self.dataset_path = dataset_path
        super().__init__(ann_file="", pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        indices_offset_map = {
            "CODE_128": 0,  # [0, 106]   total 107
            "CODE_39": 107,  # [107, 151] total 45    inverse的起止符要用另一个来表示
            "CODE_93": 152,  # [152, 200] total 49
            "CODE_EAN_UPC": 201,  # [201, 233] total 33
        }

        data_list = []
        for sub in self.dataset_path:
            all_data = os.listdir(sub)
            for folder in all_data:
                info = {}
                folder_name = os.path.join(sub, folder)
                info["img_path"] = os.path.join(folder_name, "code_img.png")
                label_path = os.path.join(folder_name, "code_info.json")
                with open(label_path, "r") as json_file:
                    label_data = json.load(json_file)
                    info["code_type"] = label_data["code_type"]

                    indices_offset = indices_offset_map[label_data["code_type"]]
                    indices = np.array(label_data["indices"], dtype=np.int32)
                    indices += indices_offset
                    info["indices"] = indices
                    info["corners"] = np.array(label_data["corners"], dtype=np.float32)
                    info["indices_len"] = len(indices)
                    data_list.append(info)

        return data_list
