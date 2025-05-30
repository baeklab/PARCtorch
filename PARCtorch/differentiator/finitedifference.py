# Differentiator/FiniteDifference.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PARCtorch.utilities.load import resolve_device


class FiniteDifference(nn.Module):
    """
    Computes spatial derivatives using finite difference filters.

    Args:
        channel_size (int): Number of input channels.
        filter_1d (np.array): 1D filter for finite difference (e.g., np.array([-1.0, 1.0])).
        padding_mode (str): Padding mode. Default value set to reproduce torchmetrics.functional.image.image_gradients.
        device (str, optional): The device to store the filters in. Default is 'cuda'.
        right_bottom (bool, optional): Whether to pad more to the right/bottom (True) or left/top (False). No effect if the padding is symmetric (odd-sized filters). Default is True.
    """

    def __init__(
        self,
        channel_size=1,
        filter_1d=np.array([-1.0, 1.0]),
        padding_mode="replicate",
        device=None,
        right_bottom=True,
    ):
        super(FiniteDifference, self).__init__()

        self.padding_mode = padding_mode
        self.channel_size = channel_size
        self.filter_size = len(filter_1d)
        self.filter_1d = filter_1d
        self.device = resolve_device(device)

        # Determine padding for dy and dx based on filter size
        # TensorFlow's padding:
        # if filter_size is odd: pad_size on both sides
        # if even: pad_size on left/top, pad_size+1 on right/bottom
        if self.filter_size % 2 == 1:
            pad_top = pad_bottom = (self.filter_size - 1) // 2
            pad_left = pad_right = (self.filter_size - 1) // 2
        elif right_bottom:
            pad_top = (self.filter_size - 1) // 2
            pad_bottom = pad_top + 1
            pad_left = (self.filter_size - 1) // 2
            pad_right = pad_left + 1
        else:
            pad_bottom = (self.filter_size - 1) // 2
            pad_top = pad_bottom + 1
            pad_right = (self.filter_size - 1) // 2
            pad_left = pad_right + 1

        self.dy_pad = (0, 0, pad_top, pad_bottom)  # (left, right, top, bottom)
        self.dx_pad = (pad_left, pad_right, 0, 0)  # (left, right, top, bottom)

        # Initialize dy_conv weights
        dy_filter = torch.tensor(self.filter_1d, dtype=torch.float32).view(
            1, 1, self.filter_size, 1
        )
        self.register_buffer(
            "dy_filter", dy_filter.repeat(channel_size, 1, 1, 1).to(self.device)
        )  # [C,1,filter_size,1]

        # Initialize dx_conv weights
        dx_filter = torch.tensor(self.filter_1d, dtype=torch.float32).view(
            1, 1, 1, self.filter_size
        )
        self.register_buffer(
            "dx_filter", dx_filter.repeat(channel_size, 1, 1, 1).to(self.device)
        )  # [C,1,1,filter_size]

    def forward(self, x):
        """
        Forward pass to compute spatial derivatives.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W]

        Returns:
            dy (torch.Tensor): Derivative along y-axis, shape [N, C, H, W]
            dx (torch.Tensor): Derivative along x-axis, shape [N, C, H, W]
        """
        # Compute dy: vertical derivative
        x_padded_dy = F.pad(
            x, self.dy_pad, mode=self.padding_mode
        )  # [N, C, H + pad_top + pad_bottom, W]
        dy = F.conv2d(x_padded_dy, self.dy_filter)

        # Compute dx: horizontal derivative
        x_padded_dx = F.pad(
            x, self.dx_pad, mode=self.padding_mode
        )  # [N, C, H, W + pad_left + pad_right]
        dx = F.conv2d(x_padded_dx, self.dx_filter)

        return dy, dx
