# Differentiator/Advection.py

import torch
import torch.nn as nn


class Advection(nn.Module):
    """
    Computes the advection term based on the state variable and velocity field.

    Args:
        finite_difference_method (nn.Module): Numerical method to calculate spatial deriviatives
    """

    def __init__(self, finite_difference_method):
        super(Advection, self).__init__()
        self.cdiff = finite_difference_method

    def forward(self, state_variable, velocity_field):
        """
        Forward pass to compute advection.

        Args:
            state_variable (torch.Tensor): Tensor of shape [N, C, H, W]
            velocity_field (torch.Tensor): Tensor of shape [N, 2C, H, W]

        Returns:
            advect (torch.Tensor): Advection term of shape [N, 1, H, W]
        """
        dy, dx = self.cdiff(state_variable)  # Each of shape [N, 1, H, W]
        spatial_deriv = torch.cat(
            [dx, dy],
            dim=1,
            # [dy, dx], dim=1
        )  # Concatenate along channel dimension: [N, 2, H, W]
        advect = torch.sum(
            spatial_deriv * velocity_field, dim=1, keepdim=True
        )  # [N, 1, H, W]
        return advect


class AdvectionUpwind(nn.Module):
    """
    Computes the advection term based on the state variable and velocity field, using upwind scheme.
    When velocity is positive, the left deriviative will be used. When velocity is negative, the right deriviative will be used.

    Args:
        left_deriviative (nn.Module): Numerical method to calculate left deriviatives
        right_deriviative (nn.Module): Numerical method to calculate right deriviatives
    """

    def __init__(self, left_deriviative, right_deriviative):
        super(AdvectionUpwind, self).__init__()
        self.ldiff = left_deriviative
        self.rdiff = right_deriviative

    def forward(self, state_variable, velocity_field):
        """
        Forward pass to compute advection.

        Args:
            state_variable (torch.Tensor): Tensor of shape [N, 1, H, W]
            velocity_field (torch.Tensor): Tensor of shape [N, 2, H, W]

        Returns:
            advect (torch.Tensor): Advection term of shape [N, 1, H, W]
        """
        dy_l, dx_l = self.ldiff(state_variable)  # Each of shape [N, 1, H, W]
        dy_r, dx_r = self.rdiff(state_variable)
        # Finding upwind
        dx = torch.where(velocity_field[:, 0:1, :, :] > 0.0, dx_l, dx_r)
        dy = torch.where(velocity_field[:, 1:2, :, :] > 0.0, dy_l, dy_r)
        spatial_deriv = torch.cat(
            [dx, dy],
            dim=1,
        )  # Concatenate along channel dimension: [N, 2, H, W]
        advect = torch.sum(
            spatial_deriv * velocity_field, dim=1, keepdim=True
        )  # [N, 1, H, W]
        return advect
