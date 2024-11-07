"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-10-09 12:07:53
Copyright: Copyright (c) 2022
LastEditTime: 2024-10-09 12:07:54
"""

import json
import os
from typing import List, Sequence

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class BarcodeDataset(BaseDataset):
    def __init__(
        self,
        ann_file_list: List[dict],
        barcode_type: str = "2d",
        pipeline: Sequence = ...,
    ):
        self.ann_file_list = ann_file_list
        barcode_1d_class_map = {"neg": 0, "bar": 1, "oned": 1, "pdf417": 2}
        barcode_2d_class_map = {"neg": 0, "dm": 1, "qr": 2, "mqr": 3}
        self.barcode_map = barcode_2d_class_map
        if barcode_type == "1d":
            self.barcode_map = barcode_1d_class_map

        super().__init__(ann_file="", pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        count = 0
        data_list = []
        for file_path_info in self.ann_file_list:
            root = file_path_info["root"]
            ann_file_path = file_path_info["path"]
            with open(ann_file_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    info = {}
                    img_path = os.path.join(root, line).split("\n")[0]
                    file_name, _ = os.path.splitext(img_path)
                    json_path = file_name + ".json"

                    with open(json_path, "r") as json_file:
                        data = json.load(json_file)
                        json_label = data["shapes"][0]["label"]
                        if json_label.lower() not in self.barcode_map:
                            continue
                        gt_label = self.barcode_map[json_label]

                    info["img_path"] = img_path
                    info["gt_label"] = gt_label
                    data_list.append(info)

                    count += 1
                    if count % 10000 == 0:
                        print("dataset count", count)

        return data_list


@DATASETS.register_module()
class BarcodeDatasetLabelFolder(BaseDataset):
    def __init__(
        self,
        img_file_paths: List,
        barcode_type: str = "2d",
        pipeline: Sequence = ...,
    ):
        self.file_paths = img_file_paths
        barcode_1d_class_map = {"neg": 0, "bar": 1, "oned": 1, "pdf417": 2}
        barcode_2d_class_map = {"neg": 0, "dm": 1, "qr": 2, "mqr": 3}
        self.barcode_map = barcode_2d_class_map
        if barcode_type == "1d":
            self.barcode_map = barcode_1d_class_map

        super().__init__(ann_file="", pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        data_list = []
        for path in self.file_paths:
            label_name = str(path).split("/")[-1]
            gt_label = -1
            if label_name.lower() in self.barcode_map:
                gt_label = self.barcode_map[label_name.lower()]

            for root, dirs, files in os.walk(path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                        img_path = os.path.join(root, file)

                        info = {}
                        info["img_path"] = img_path
                        info["gt_label"] = gt_label
                        data_list.append(info)

        return data_list
