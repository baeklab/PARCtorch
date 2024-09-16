# Differentiator/Advection.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FiniteDifference import FiniteDifference

class Advection(nn.Module):
    """
    Computes the advection term based on the state variable and velocity field.
    
    Args:
        channel_size (int): Number of input channels for the state variable.
        cd_filter_1d (np.array): 1D filter for finite difference (e.g., np.array([-1.0, 1.0])).
        padding_mode (str): Padding mode. "SYMMETRIC" in TensorFlow corresponds to "reflect" in PyTorch.
    """
    def __init__(self, channel_size=1, cd_filter_1d=np.array([-1.0, 1.0]), padding_mode="SYMMETRIC"):
        super(Advection, self).__init__()
        self.cdiff = FiniteDifference(channel_size, cd_filter_1d, padding_mode)

    def forward(self, state_variable, velocity_field):
        """
        Forward pass to compute advection.
        
        Args:
            state_variable (torch.Tensor): Tensor of shape [N, C, H, W]
            velocity_field (torch.Tensor): Tensor of shape [N, 2C, H-1, W-1]
        
        Returns:
            advect (torch.Tensor): Advection term of shape [N, 1, H-1, W-1]
        """
        dy, dx = self.cdiff(state_variable)  # Each of shape [N, C, H-1, W-1]
        spatial_deriv = torch.cat([dx, dy], dim=1)  # Concatenate along channel dimension: [N, 2C, H-1, W-1]
        advect = torch.sum(spatial_deriv * velocity_field, dim=1, keepdim=True)  # [N, 1, H-1, W-1]
        return advect
