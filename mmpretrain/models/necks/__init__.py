"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-08 14:02:30
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-08 14:02:30
"""

# Copyright (c) OpenMMLab. All rights reserved.
from .beitv2_neck import BEiTV2Neck
from .cae_neck import CAENeck
from .densecl_neck import DenseCLNeck
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .gsp import GlobalStripPooling
from .hr_fuse import HRFuseScales
from .itpn_neck import iTPNPretrainDecoder
from .linear_neck import LinearNeck
from .mae_neck import ClsBatchNormNeck, MAEPretrainDecoder
from .milan_neck import MILANPretrainDecoder
from .mixmim_neck import MixMIMPretrainDecoder
from .mocov2_neck import MoCoV2Neck
from .nonlinear_neck import NonLinearNeck
from .simmim_neck import SimMIMLinearDecoder
from .spark_neck import SparKLightDecoder
from .swav_neck import SwAVNeck

__all__ = [
    "GlobalAveragePooling",
    "GlobalStripPooling",
    "GeneralizedMeanPooling",
    "HRFuseScales",
    "LinearNeck",
    "BEiTV2Neck",
    "CAENeck",
    "DenseCLNeck",
    "MAEPretrainDecoder",
    "ClsBatchNormNeck",
    "MILANPretrainDecoder",
    "MixMIMPretrainDecoder",
    "MoCoV2Neck",
    "NonLinearNeck",
    "SimMIMLinearDecoder",
    "SwAVNeck",
    "iTPNPretrainDecoder",
    "SparKLightDecoder",
]
