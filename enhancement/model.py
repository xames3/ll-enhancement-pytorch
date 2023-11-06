"""\
Model Specifications
===================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, November 05 2023
Last updated on: Monday, November 06 2023
"""

from __future__ import annotations

import typing as t

import torch
import torch.nn as nn

from .utils import BaseEnhancementCNN
from .utils import _Tensor
from .utils import config
from .utils import validate


class DecompositionCNN(BaseEnhancementCNN):
    """..."""

    shallow_feature_layer: nn.Conv2d = {
        "layer": nn.Conv2d(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.OUTPUT_CHANNELS,
            kernel_size=config.KERNEL_SIZE,
            padding=4,
            padding_mode="replicate",
        )
    }
    activated_layers: nn.Sequential = {
        "sequence": [
            nn.Conv2d(
                in_channels=config.OUTPUT_CHANNELS,
                out_channels=config.OUTPUT_CHANNELS,
                kernel_size=config.KERNEL_SIZE,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.OUTPUT_CHANNELS,
                out_channels=config.OUTPUT_CHANNELS,
                kernel_size=config.KERNEL_SIZE,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.OUTPUT_CHANNELS,
                out_channels=config.OUTPUT_CHANNELS,
                kernel_size=config.KERNEL_SIZE,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.OUTPUT_CHANNELS,
                out_channels=config.OUTPUT_CHANNELS,
                kernel_size=config.KERNEL_SIZE,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.OUTPUT_CHANNELS,
                out_channels=config.OUTPUT_CHANNELS,
                kernel_size=config.KERNEL_SIZE,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
        ]
    }
    reconstructed_layer: nn.Conv2d = {
        "layer": nn.Conv2d(
            in_channels=config.OUTPUT_CHANNELS,
            out_channels=config.INPUT_CHANNELS,
            kernel_size=config.KERNEL_SIZE,
            padding=1,
            padding_mode="replicate",
        )
    }

    @validate
    def forward(self, x: _Tensor) -> t.Any:
        """Implementation of forward pass."""
        max_input = torch.max(x, dim=1, keepdim=True)[0]
        input_img = torch.cat((max_input, x), dim=1)
        shallow_features = self.shallow_feature_layer(input_img)
        activated_features = self.activated_layers(shallow_features)
        output_features = self.reconstructed_layer(activated_features)
        R = torch.sigmoid(output_features[:, 0:3, :, :])
        L = torch.sigmoid(output_features[:, 3:4, :, :])
        return R, L
