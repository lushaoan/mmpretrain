"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-12 17:17:57
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-12 17:17:58
"""

"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 18:58:07
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 18:58:08
"""

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class FocalCTCLoss(nn.Module):
    def __init__(self, blank: int, alpha=0.25, gamma=2.0, reduction="mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ctc = torch.nn.CTCLoss(blank, reduction="none", zero_infinity=True)

    def focal_enabled(self):
        return self.alpha > 0 and self.gamma >= 0

    def forward(
        self,
        tokens_btc: torch.Tensor,
        targets_bxt: torch.Tensor,
        targets_lengths_bx1: torch.Tensor,
    ) -> torch.Tensor:
        # [b t c] ==> [t b c]
        tokens_tbc = tokens_btc.swapdims(0, 1)

        # compute log prob on axis `c`
        logits_tbc = tokens_tbc.log_softmax(2)
        logits_tbc = logits_tbc.cpu().contiguous()

        seq_len = logits_tbc.size(0)

        input_lengths_nx1 = torch.full_like(targets_lengths_bx1, seq_len)

        loss = self.ctc(
            logits_tbc,
            targets_bxt,
            input_lengths_nx1,
            targets_lengths_bx1,
        )

        if self.focal_enabled():
            # FL(pt) = -alpha*[(1-pt)^gamma] * log(pt)
            # here we use pt = exp(-x), so log(pt) == log(exp(-x)) == -x
            # then we have: FL(pt) = -alpha*[(1-pt)^gamma]*(-x)
            # ==> FL(pt) = alpha*[(1-pt)^gamma]*x
            pt = torch.exp(-loss)
            loss = self.alpha * ((1 - pt) ** self.gamma) * loss

        if self.reduction == "mean":
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)

        return loss
