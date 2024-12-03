"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-07 14:18:44
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-07 14:18:45
"""

import copy
from typing import Callable, NamedTuple, Tuple, Union

import torch
from mmengine.model import BaseModule, Sequential
from torch import nn

from mmpretrain.models.utils import make_divisible
from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


class PointWiseConv(Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias=True,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()

        bias = False if norm_layer is not None else bias

        self.add_module(
            "pw_conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
        )

        if norm_layer is not None:
            self.add_module("norm", norm_layer(out_channels))

        if act_layer is not None:
            self.add_module("act", act_layer())


class ConvNormAct(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        with_bn=True,
        with_act=True,
    ) -> None:
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self._norm = nn.BatchNorm2d(out_channels) if with_bn else nn.Identity()
        self._act = nn.ReLU6() if with_act else nn.Identity()

    def forward(self, x_nchw):
        x = self._conv(x_nchw)
        x = self._norm(x)
        x = self._act(x)

        return x


class ConcatTensor(BaseModule):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim

    def forward(self, tensors):
        return torch.cat(tensors, dim=self.dim)


class IsotropicDwConv(Sequential):
    def __init__(
        self,
        channels,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias=True,
        norm_layer=None,
        act_layer=None,
    ) -> None:
        super().__init__()

        bias = False if norm_layer is not None else bias

        self.add_module(
            "dw_conv",
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=bias,
            ),
        )

        if norm_layer is not None:
            self.add_module("norm", norm_layer(channels))

        if act_layer is not None:
            self.add_module("act", act_layer())


class ChannelScaling(BaseModule):
    """Scale vector by element multiplications."""

    def __init__(self, dim: int, init_value=1.0, trainable=True) -> None:
        super().__init__()

        # use depth-wise 1x1 conv to implement channel-wise multiplication
        self.scaling_factors = nn.Conv2d(
            dim, dim, kernel_size=1, groups=dim, bias=False
        )

        # init weights
        self.scaling_factors.weight.requires_grad_(trainable)
        nn.init.constant_(self.scaling_factors.weight, init_value)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        out = self.scaling_factors(x_nchw)
        return out


class MLP(BaseModule):
    """Linear output bottleneck MLP module."""

    def __init__(self, in_chan, out_chan, hidden_expand_ratio=3.0) -> None:
        super().__init__()

        hidden_chan = make_divisible(out_chan * hidden_expand_ratio, divisor=8)

        # no expansion, use single fc layer only
        if hidden_chan == out_chan:
            self._mlp = PointWiseConv(
                in_chan,
                hidden_chan,
                bias=False,
                norm_layer=nn.BatchNorm2d,
                act_layer=None,
            )
        else:
            self._mlp = nn.Sequential(
                PointWiseConv(
                    in_chan,
                    hidden_chan,
                    bias=False,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.ReLU6,
                ),
                PointWiseConv(
                    hidden_chan,
                    out_chan,
                    bias=False,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=None,
                ),
            )

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.fc2(x)

        x = self._mlp(x)

        return x


class TokenMixer(BaseModule):
    r"""Base class for token-mixer.

    Note:
        The token-mixer shouldn't contains any normalization
    and activation layer!!!

    Args:
        dim: channel of input feature map
    """

    def __init__(self, dim: int) -> None:
        assert isinstance(dim, int) and dim > 0

        super().__init__()

        self._dim = dim

    @property
    def out_channels(self) -> int:
        return self._dim


class InceptionConcatDWConvTokenMixer(TokenMixer):
    def __init__(
        self,
        dim: int,
        res_connect=True,
        res_scale=False,
        square_kernel_size: int = 3,
        dw_conv_w: int = 11,
        dw_conv_h: int = 11,
    ) -> None:
        super().__init__(dim)

        assert square_kernel_size % 2, "expect odd square_kernel_size "
        assert dw_conv_w % 2, "expect odd dw_conv_w"
        assert dw_conv_h % 2, "expect odd dw_conv_h"
        assert (
            dim % 4 == 0
        ), "expect input channel can be divided by 4. and now channel is {}. ".format(
            dim
        )
        self.res_connect = res_connect
        if res_scale:
            self.residual = nn.Sequential(
                PointWiseConv(
                    dim,
                    dim // 4,
                    bias=False,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=None,
                ),
                ChannelScaling(
                    dim // 4,
                    init_value=1.0,
                    trainable=True,
                ),
                nn.BatchNorm2d(dim // 4),
            )
        else:
            self.residual = nn.Sequential(
                PointWiseConv(
                    dim,
                    dim // 4,
                    bias=False,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=None,
                ),
                nn.Identity(),
            )

        self.dw_conv_hw = nn.Sequential(
            PointWiseConv(
                dim,
                dim // 4,
                bias=False,
                norm_layer=nn.BatchNorm2d,
                act_layer=None,
            ),
            IsotropicDwConv(
                dim // 4,
                square_kernel_size,
                padding=square_kernel_size // 2,
                norm_layer=nn.BatchNorm2d,
            ),
        )
        self.dw_conv_w = nn.Sequential(
            PointWiseConv(
                dim,
                dim // 4,
                bias=False,
                norm_layer=nn.BatchNorm2d,
                act_layer=None,
            ),
            IsotropicDwConv(
                dim // 4,
                kernel_size=(1, dw_conv_w),
                padding=(0, dw_conv_w // 2),
                norm_layer=nn.BatchNorm2d,
            ),
        )
        self.dw_conv_h = nn.Sequential(
            PointWiseConv(
                dim,
                dim // 4,
                bias=False,
                norm_layer=nn.BatchNorm2d,
                act_layer=None,
            ),
            IsotropicDwConv(
                dim // 4,
                kernel_size=(dw_conv_h, 1),
                padding=(dw_conv_h // 2, 0),
                norm_layer=nn.BatchNorm2d,
            ),
        )

        self.concat = ConcatTensor(dim=1)

    @property
    def out_channels(self):
        return 4 * (self._dim // 4) if self.res_connect else 3 * (self._dim // 4)

    @property
    def norm_required(self):
        return False

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        tensors = []
        if self.res_connect:
            tensors.append(self.residual(x_nchw))

        tensors.append(self.dw_conv_hw(x_nchw))
        tensors.append(self.dw_conv_w(x_nchw))
        tensors.append(self.dw_conv_h(x_nchw))

        out = self.concat(tensors)

        return out


class ConvNeXtTokenMixer(TokenMixer):
    r"""From paper:
        A ConvNet for the 2020s
        https://arxiv.org/abs/2201.03545

    Actually, it's only a 7x7 depth-separable conv.
    """

    def __init__(self, dim, kernel_h=7, kernel_w=7) -> None:
        super().__init__(dim)

        assert kernel_h % 2 == 1, "expect odd kernel_size!"
        assert kernel_w % 2 == 1, "expect odd kernel_size!"

        self.dw_conv = IsotropicDwConv(
            dim,
            kernel_size=(kernel_h, kernel_w),
            padding=(kernel_h // 2, kernel_w // 2),
            bias=False,
        )

    @property
    def norm_required(self):
        return True

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        out = self.dw_conv(x_nchw)

        return out


def get_token(token_type: str):
    mixers = {
        "ConvNeXtTokenMixer": ConvNeXtTokenMixer,
        "InceptionConcatDWConvTokenMixer": InceptionConcatDWConvTokenMixer,
    }

    return mixers[token_type]


class MetaNeXtBlock(BaseModule):
    r"""Building block for multi-stage backbone. The block comes from paper:

        InceptionNeXt: When Inception Meets ConvNeXt, section 3.1 MetaNeXt.

        https://arxiv.org/abs/2303.16900.

    The architecture of block is :

    input -> token_mixer -> BN -> FC -> ReLU6 -> FC -> + output
          |                                            ^
          v--------------------------------------------|

    First Fully-Connected(FC) use to expand the channel of feature map, while
    the second use to reduce the channels back so that we can add shortcut.
    Theses two fc layers combined to a mlp with single hidden layer.

    Both first and second FC layers can implemented by linear layer or 1x1
    point-wise conv. We choose 1x1 conv here, to keep the block a pure-conv block.

    Args:
        dim: channels of the input feature map
        expand_ratio: expand ratio in channel dimension.
        token_mixer_builder: factory method use to build the token-mixer.
            The factory accepts an int parameter which describes the dimension of
            input feature map, and return a nn.Module(or it's subclass) object.
    """

    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        token_mixer_setting: dict,
        norm_layer: Callable[[int], nn.Module],
    ) -> None:
        assert isinstance(dim, int) and dim >= 1, dim
        assert expand_ratio > 0, expand_ratio

        super().__init__()

        self.expand_ratio = expand_ratio

        token_cls = get_token(token_mixer_setting["type"])
        token_settting = copy.deepcopy(token_mixer_setting)
        del token_settting["type"]
        self.token_mixer: TokenMixer = token_cls(dim, **token_settting)
        if self.token_mixer.norm_required:
            self.norm = norm_layer(self.token_mixer.out_channels)
        else:
            self.norm = nn.Identity()

        self.mlp = MLP(self.token_mixer.out_channels, dim, expand_ratio)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        shortcut = x_nchw
        x = self.token_mixer(x_nchw)
        x = self.norm(x)
        x = self.mlp(x)
        out = x + shortcut

        return out


class ArchSetting(NamedTuple):
    num_blocks: int
    channels: int
    expand_ratio: Union[float, int]
    stride_hw: Union[tuple, list]
    token_mixer_setting: dict


@MODELS.register_module()
class MultiStageBackbone(BaseBackbone):
    def __init__(self, **kwargs):
        super(MultiStageBackbone, self).__init__()
        self.in_channels = kwargs["in_channels"]
        # self.token_mixer = self.kwargs["token_mixer"]
        self.arch_settings = [ArchSetting(*s) for s in kwargs["arch_settings"]]

        pre_stage_channels = self.in_channels
        stages = []
        for setting in self.arch_settings:
            stride_hw = setting.stride_hw
            stage = nn.Sequential()

            if stride_hw == (1, 1) and pre_stage_channels == setting.channels:
                downsample = nn.Identity()
            else:
                downsample = ConvNormAct(
                    pre_stage_channels,
                    setting.channels,
                    kernel_size=3,
                    stride=stride_hw,
                    padding=1,
                )

            stage.add_module("adapt_layer", downsample)

            for i in range(setting.num_blocks):
                block = MetaNeXtBlock(
                    setting.channels,
                    setting.expand_ratio,
                    setting.token_mixer_setting,
                    nn.BatchNorm2d,
                )

                stage.add_module(f"block_{i}", block)

            stages.append(stage)
            pre_stage_channels = setting.channels

        self.stages = nn.ModuleList(stages)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # shape of x: [B, C, H, W]
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features
