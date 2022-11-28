from typing import Tuple
from functools import partial

import torch
import torch.nn as nn

from compressai.layers import AttentionBlock, ResidualBlock
from nip import nip


def _conv5x5(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=5,
                     padding=2,
                     stride=2)


def _conv5x5_transpose(in_channels: int, out_channels: int) -> nn.Module:
    return nn.ConvTranspose2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=5,
                              padding=2,
                              stride=2,
                              output_padding=1)


def _residual_blocks(in_channels: int, n_blocks: int) -> nn.Module:
    return nn.Sequential(*[ResidualBlock(in_ch=in_channels,
                                         out_ch=in_channels) for _ in range(n_blocks)])


@nip
class ConvEncoder(nn.Module):

    def __init__(self,
                 channels: Tuple[int, ...],
                 in_channels: int = 3,
                 n_residual_blocks: int = 3):
        super(ConvEncoder, self).__init__()
        self._in_channels = in_channels
        self._out_channels = channels[-1]

        residual_factory = partial(_residual_blocks, n_blocks=n_residual_blocks)

        layers = []
        all_channels = (in_channels,) + channels
        for i in range(len(all_channels) - 1):
            c1 = all_channels[i]
            c2 = all_channels[i+1]
            layers.append(_conv5x5(c1, c2))
            is_last = i == len(all_channels) - 2
            if not is_last:
                layers.append(residual_factory(in_channels=c2))
            if (i % 2) != 0 or is_last:
                layers.append(AttentionBlock(c2))

        self._layers = nn.Sequential(*layers)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers.forward(x)


@nip
class ConvDecoder(nn.Module):

    def __init__(self,
                 channels: Tuple[int, ...],
                 out_channels: int = 3,
                 n_residual_blocks: int = 3):
        super(ConvDecoder, self).__init__()
        self._in_channels = channels[0]
        self._out_channels = out_channels

        residual_factory = partial(_residual_blocks, n_blocks=n_residual_blocks)

        layers = [AttentionBlock(channels[0])]
        all_channels = channels + (out_channels,)
        for i in range(len(all_channels) - 1):
            c1 = all_channels[i]
            c2 = all_channels[i + 1]
            layers.append(_conv5x5_transpose(c1, c2))
            is_last = i == len(all_channels) - 2
            if (i % 2) != 0 and not is_last:
                layers.append(AttentionBlock(c2))
            if not is_last:
                layers.append(residual_factory(in_channels=c2))

        self._layers = nn.Sequential(*layers)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers.forward(x)
