"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-14 10:16:22
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-14 10:16:22
"""

from typing import Sequence

import editdistance
import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from mmpretrain.models.heads.ctc_head import CTCHead


def ToList(obj) -> list:
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], int)):
        return obj

    if isinstance(obj, np.ndarray):
        return obj.astype("int").tolist()

    if isinstance(obj, torch.Tensor):
        return obj.to(dtype=torch.int).tolist()

    return list(int(i) for i in obj)


@METRICS.register_module()
class OnedDecoderMetric(BaseMetric):
    def __init__(
        self,
        index_offset=1,
        collect_device: str = "cpu",
        prefix: str = None,
        collect_dir: str = "cpu",
    ) -> None:
        super().__init__(collect_device, prefix, collect_dir)
        self.index_offset = index_offset

        self._total_eval_samples = 0
        self._sum_ed_total = 0
        self._sum_seq_len_total = 0
        self._sum_decoded = 0
        self._eval_batch_count = 0

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        sum_ed = 0
        sum_seq_len = 0
        num_decoded = 0
        batch_size = len(data_samples)
        for i in range(0, batch_size):
            target_dict = data_batch["data_samples"][i]
            predict_c1t = data_samples[i]

            target_indices = target_dict.indices
            target_indices_len = target_dict.indices_len
            target_indices = target_indices[:target_indices_len]

            predicts_cxt = predict_c1t.squeeze(dim=1)
            predicts_txc = torch.transpose(predicts_cxt, 0, 1)
            predicted_indices = CTCHead.decode_result(
                predicts_txc=predicts_txc, index_offset=self.index_offset
            )

            predicted_indices = ToList(predicted_indices)
            target_indices = ToList(target_indices)
            edit_dist = editdistance.eval(predicted_indices, target_indices)

            sum_ed += edit_dist
            sum_seq_len += target_indices_len.detach().cpu().item()
            num_decoded += edit_dist == 0

        self._sum_ed_total += sum_ed
        self._sum_decoded += num_decoded
        self._sum_seq_len_total += sum_seq_len
        self._total_eval_samples += batch_size

    def compute_metrics(self, results: list) -> dict:
        index_error_rate = self._sum_ed_total / (self._sum_seq_len_total + 1e-8)
        decode_rate = self._sum_decoded / (self._total_eval_samples + 1e-8)

        self._sum_ed_total = 0
        self._sum_decoded = 0
        self._sum_seq_len_total = 0
        self._total_eval_samples = 0

        result_metrics = dict()
        result_metrics["index_error_rate"] = index_error_rate
        result_metrics["decode_rate"] = decode_rate

        return result_metrics
