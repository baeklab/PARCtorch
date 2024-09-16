# Differentiator/Diffusion.py

import torch
import torch.nn as nn
import numpy as np
from .FiniteDifference import FiniteDifference  # Correct relative import

class Diffusion(nn.Module):
    """
    Computes the Laplacian of the state variable using finite difference filters.
    
    Args:
        channel_size (int): Number of input channels.
        cd_filter_1d (np.array): 1D filter for finite difference (e.g., np.array([-1.0, 1.0])).
        padding (str): Padding mode. "SYMMETRIC" in TensorFlow corresponds to "reflect" in PyTorch.
    """
    def __init__(self, channel_size=1, cd_filter_1d=np.array([-1.0, 1.0]), padding="SYMMETRIC"):
        super(Diffusion, self).__init__()
        self.cdiff = FiniteDifference(channel_size, cd_filter_1d, padding)

    def forward(self, state_variable):
        """
        Forward pass to compute the Laplacian of the state variable.
        
        Args:
            state_variable (torch.Tensor): Tensor of shape [N, C, H, W]
        
        Returns:
            laplacian (torch.Tensor): Laplacian of shape [N, C, H-2, W-2] (for filter_size=2)
        """
        dy, dx = self.cdiff(state_variable)     # First derivatives
        dyy, _ = self.cdiff(dy)                # Second derivative w.r.t y
        _, dxx = self.cdiff(dx)                # Second derivative w.r.t x
        laplacian = dyy + dxx                   # Sum of second derivatives
        return laplacian
