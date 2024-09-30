# Integrator/Poisson.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from PARCtorch.utilities.resnet import ResNet


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
        uy, ux = self.cdiff(vector_field[:, 0:1, :, :])
        vy, vx = self.cdiff(vector_field[:, 1:2, :, :])
        ux2 = ux * ux
        vy2 = vy * vy
        uyvx = uy * vx

        return ux2, vy2, uyvx


class PoissonBlock(nn.Module):
    """
    Poisson Block to compute pressure terms from input vector fields using existing UnetDownBlock and ResNetBlock.

    Args:
        n_input_channel (int): Number of input channels. We always assume the last 2 channels will be the one to run the Poisson operator on.
        finite_difference_method (nn.Module): Numerical method for finite difference.
        kernel_size (int, optional): Kernel size. Default is 3.
        n_base_features (int, optional): Number of channels of the convolutional layers. Default is 64.
        padding_mode (str, optional): Padding mode for the convolutional layers. Default is "constant".
    """

    def __init__(
        self,
        n_input_channel: int,
        finite_difference_method: nn.Module,
        kernel_size: int = 3,
        n_base_features: int = 64,
        padding_mode: str = "constant",
    ):
        super(PoissonBlock, self).__init__()

        self.padding_mode = padding_mode
        self.pad_size = (kernel_size - 1) // 2

        # Initialize the Poisson layer
        self.poisson = Poisson(finite_difference_method)
        # Initialize other layers
        self.conv = nn.Sequential(
            nn.Conv2d(3 + n_input_channel, n_base_features, kernel_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_base_features, n_base_features, 1),
            nn.LeakyReLU(0.2),
            ResNet(
                n_base_features,
                [n_base_features, n_base_features],
                kernel_size,
                False,
                padding_mode,
            ),
            nn.Conv2d(n_base_features, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute pressure terms.

        Args:
            x (torch.Tensor): Input tensor of shape [N, n_input_channel, y, x].

        Returns:
            torch.Tensor: Output tensor of shape [N, 1, y, x].
        """
        out = self.poisson(x[:, -2:, :, :])
        out = torch.cat(out, 1)
        out = torch.cat([x, out], 1)
        out = F.pad(
            out,
            pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode=self.padding_mode,
            value=0,
        )
        out = self.conv(out)
        return out
