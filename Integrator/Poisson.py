# Integrator/Poisson.py

import torch
import torch.nn as nn
import numpy as np
from Differentiator.FiniteDifference import FiniteDifference  # Absolute import from Differentiator package

class Poisson(nn.Module):
    """
    Computes Poisson terms based on the vector field using finite difference filters.
    
    Args:
        channel_size (int): Number of input channels for each vector component.
        cd_filter_1d (np.array): 1D filter for finite difference (e.g., np.array([-1.0, 1.0])).
        padding (str): Padding mode. "SYMMETRIC" in TensorFlow corresponds to "reflect" in PyTorch.
    """
    def __init__(self, finite_difference_method):
        super(Poisson, self).__init__()
        self.cdiff = finite_difference_method

    def forward(self, vector_field):
        """
        Forward pass to compute Poisson terms from the vector field.
        
        Args:
            vector_field (torch.Tensor): Tensor of shape [N, 2, H, W].
        
        Returns:
            ux2 (torch.Tensor): Element-wise square of the x-derivative of the first component, shape [N, C, H, W]
            vy2 (torch.Tensor): Element-wise square of the y-derivative of the second component, shape [N, C, H, W]
            uyvx (torch.Tensor): Element-wise product of the y-derivative of the first component and the x-derivative of the second component, shape [N, C, H, W]
        """
        # Compute derivatives for the first component
        uy, ux = self.cdiff(vector_field[:, 0:1, :, :])  # uy: d/dy of first component, ux: d/dx of first component
        
        # Compute derivatives for the second component
        vy, vx = self.cdiff(vector_field[:, 1:2, :, :])  # vy: d/dy of second component, vx: d/dx of second component
        
        # Compute Poisson terms
        ux2 = ux * ux       # Element-wise square of ux
        vy2 = vy * vy       # Element-wise square of vy
        uyvx = uy * vx      # Element-wise product of uy and vx
        
        return ux2, vy2, uyvx
