# Differentiator/FiniteDifference.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FiniteDifference(nn.Module):
    """
    Computes spatial derivatives using finite difference filters.
    
    Args:
        channel_size (int): Number of input channels.
        filter_1d (np.array): 1D filter for finite difference (e.g., np.array([-1.0, 1.0])).
        padding (str): Padding mode. "SYMMETRIC" in TensorFlow corresponds to "reflect" in PyTorch.
                      Other options include "constant", "replicate", "reflect", etc.
    """
    def __init__(self, channel_size=1, filter_1d=np.array([-1.0, 1.0]), padding="SYMMETRIC"):
        super(FiniteDifference, self).__init__()
        
        # Map TensorFlow padding to PyTorch padding modes
        padding_map = {
            "SYMMETRIC": "reflect",
            "CONSTANT": "constant",
            "REFLECT": "reflect",
            "REPLICATE": "replicate",
        }
        self.padding_mode = padding_map.get(padding.upper(), "constant")
        self.channel_size = channel_size
        self.filter_size = len(filter_1d)
        self.filter_1d = filter_1d

        # Determine padding for dy and dx based on filter size
        # TensorFlow's padding:
        # if filter_size is odd: pad_size on both sides
        # if even: pad_size on left/top, pad_size+1 on right/bottom
        if self.filter_size % 2 == 1:
            pad_top = pad_bottom = (self.filter_size - 1) // 2
            pad_left = pad_right = (self.filter_size - 1) // 2
        else:
            pad_top = (self.filter_size - 1) // 2
            pad_bottom = pad_top + 1
            pad_left = (self.filter_size - 1) // 2
            pad_right = pad_left + 1

        self.dy_pad = (0, 0, pad_top, pad_bottom)  # (left, right, top, bottom)
        self.dx_pad = (pad_left, pad_right, 0, 0)  # (left, right, top, bottom)

        # Define convolutional layers for dy and dx
        # dy_conv: vertical difference
        self.dy_conv = nn.Conv2d(in_channels=channel_size,
                                 out_channels=channel_size,
                                 kernel_size=(self.filter_size, 1),
                                 padding=0,
                                 bias=False,
                                 groups=channel_size)  # Depthwise convolution
        
        # dx_conv: horizontal difference
        self.dx_conv = nn.Conv2d(in_channels=channel_size,
                                 out_channels=channel_size,
                                 kernel_size=(1, self.filter_size),
                                 padding=0,
                                 bias=False,
                                 groups=channel_size)  # Depthwise convolution
        
        # Initialize the weights for dy_conv and dx_conv
        with torch.no_grad():
            # Initialize dy_conv weights
            dy_filter = torch.tensor(self.filter_1d, dtype=torch.float32).view(1, 1, self.filter_size, 1)
            dy_filter = dy_filter.repeat(channel_size, 1, 1, 1)  # [C,1,filter_size,1]
            self.dy_conv.weight.copy_(dy_filter)
            
            # Initialize dx_conv weights
            dx_filter = torch.tensor(self.filter_1d, dtype=torch.float32).view(1, 1, 1, self.filter_size)
            dx_filter = dx_filter.repeat(channel_size, 1, 1, 1)  # [C,1,1,filter_size]
            self.dx_conv.weight.copy_(dx_filter)

    def forward(self, x):
        """
        Forward pass to compute spatial derivatives.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W]
        
        Returns:
            dy (torch.Tensor): Derivative along y-axis, shape [N, C, H-1, W-1] (for filter_size=2)
            dx (torch.Tensor): Derivative along x-axis, shape [N, C, H-1, W-1] (for filter_size=2)
        """
        # Compute dy: vertical derivative
        x_padded_dy = F.pad(x, self.dy_pad, mode=self.padding_mode)  # [N, C, H + pad_top + pad_bottom, W]
        dy = self.dy_conv(x_padded_dy)  # [N, C, H_new, W]
        
        # Compute dx: horizontal derivative
        x_padded_dx = F.pad(x, self.dx_pad, mode=self.padding_mode)  # [N, C, H, W + pad_left + pad_right]
        dx = self.dx_conv(x_padded_dx)  # [N, C, H, W_new]
        
        # Align spatial dimensions by cropping
        if self.filter_size % 2 == 0:
            # Even filter size: crop last row and column
            dy = dy[:, :, :-1, :-1]  # [N, C, H-1, W-1]
            dx = dx[:, :, :-1, :-1]  # [N, C, H-1, W-1]
        else:
            # Odd filter size: crop last row and column for consistency
            dy = dy[:, :, :-1, :-1]  # [N, C, H-1, W-1]
            dx = dx[:, :, :-1, :-1]  # [N, C, H-1, W-1]
        
        return dy, dx
