# Differentiator/Diffusion.py

import torch.nn as nn


class Diffusion(nn.Module):
    """
    Computes the Laplacian of the state variable using finite difference filters.

    Args:
        finite_difference_method (nn.Module): Numerical method to calculate spatial deriviatives
    """

    def __init__(self, finite_difference_method):
        super(Diffusion, self).__init__()
        self.cdiff = finite_difference_method

    def forward(self, state_variable):
        """
        Forward pass to compute the Laplacian of the state variable.

        Args:
            state_variable (torch.Tensor): Tensor of shape [N, C, H, W]

        Returns:
            laplacian (torch.Tensor): Laplacian of shape [N, C, H, W]
        """
        dy, dx = self.cdiff(state_variable)  # First derivatives
        dyy, _ = self.cdiff(dy)  # Second derivative w.r.t y
        _, dxx = self.cdiff(dx)  # Second derivative w.r.t x
        laplacian = dyy + dxx  # Sum of second derivatives
        return laplacian
