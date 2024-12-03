'''
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-07 19:07:14
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-07 19:08:52
'''
# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
from mmengine.utils import digit_version

from .apis import *  # noqa: F401, F403
from .version import __version__

mmcv_minimum_version = '2.0.0'
mmcv_maximum_version = '2.4.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.8.3'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

__all__ = ['__version__']
