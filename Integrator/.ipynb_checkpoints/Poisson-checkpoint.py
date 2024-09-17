# Integrator/Poisson.py

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Conv2d, ReLU
from Differentiator.FiniteDifference import FiniteDifference  # Ensure correct import path
from Utilities.Unet import UnetDownBlock
from Utilities.ResNet import ResNet  # Ensure this import path is correct based on your project structure

class Poisson(nn.Module):
    """
    Computes Poisson terms based on the vector field using finite difference filters.
    
    Args:
        channel_size (int): Number of input channels for each vector component.
        cd_filter_1d (np.ndarray): 1D filter for finite difference (e.g., np.array([-1.0, 1.0])).
        padding_mode (str): Padding mode. "SYMMETRIC" in TensorFlow corresponds to "reflect" in PyTorch.
    """
    def __init__(self, channel_size=1, cd_filter_1d=np.array([-1.0, 1.0]), padding_mode="reflect"):
        super(Poisson, self).__init__()
        self.cdiff = FiniteDifference(channel_size, cd_filter_1d, padding_mode)

    def forward(self, vector_field: tuple) -> tuple:
        """
        Forward pass to compute Poisson terms from the vector field.
        
        Args:
            vector_field (tuple of torch.Tensor): 
                - vector_field[0]: Tensor of shape [N, C, H, W] representing the first component (e.g., y-component).
                - vector_field[1]: Tensor of shape [N, C, H, W] representing the second component (e.g., x-component).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - ux2: Element-wise square of the x-derivative of the first component.
                - vy2: Element-wise square of the y-derivative of the second component.
                - uyvx: Element-wise product of the y-derivative of the first component and the x-derivative of the second component.
        """
        # Compute derivatives for the first component
        uy, ux = self.cdiff(vector_field[0])  # uy: d/dy of first component, ux: d/dx of first component
        
        # Compute derivatives for the second component
        vy, vx = self.cdiff(vector_field[1])  # vy: d/dy of second component, vx: d/dx of second component
        
        # Compute Poisson terms
        ux2 = ux * ux       # Element-wise square of ux
        vy2 = vy * vy       # Element-wise square of vy
        uyvx = uy * vx      # Element-wise product of uy and vx
        
        return ux2, vy2, uyvx

class PoissonBlock(nn.Module):
    """
    Poisson Block to compute pressure terms from input vector fields using existing UnetDownBlock and ResNetBlock.
    
    Args:
        n_base_features (int): Number of base feature channels. Default is 64.
    """
    def __init__(self, n_base_features=64):
        super(PoissonBlock, self).__init__()
        
        # Initialize the Poisson layer
        self.poisson = Poisson(
            channel_size=1, 
            cd_filter_1d=np.array([-1.0, 1.0]), 
            padding_mode="reflect"  # 'SYMMETRIC' in TensorFlow corresponds to 'reflect' in PyTorch
        )
        
        # Initialize the UnetDownBlock
        # Assuming input has 3 channels, and Poisson outputs 3 additional channels (ux2, vy2, uyvx)
        # Hence, concatenated tensor has 3 + 3 = 6 channels
        self.unet_down = UnetDownBlock(
            in_channels=6, 
            out_channels=n_base_features, 
            kernel_size=3, 
            padding_mode='reflect'
        )
        
        # Initialize the ResNet block
        # Assuming ResNet expects block_dimensions; here, using [n_base_features, n_base_features]
        self.resnet_block = ResNet(
            in_channels=n_base_features, 
            block_dimensions=[n_base_features, n_base_features], 
            kernel_size=3, 
            pooling=False, 
            padding_mode='reflect'
        )
        
        # Final Conv2d to output 1 channel (pressure term)
        self.final_conv = nn.Conv2d(
            in_channels=n_base_features, 
            out_channels=1, 
            kernel_size=1, 
            padding=0  # 'same' padding equivalent is achieved via padding in previous layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute pressure terms.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, 3, H, W].
        
        Returns:
            torch.Tensor: Output tensor of shape [N, 1, H, W].
        """
        # x: [N, 3, H, W]
        
        # Extract the first two channels for Poisson computation
        vector_field = (x[:, 0:1, :, :], x[:, 1:2, :, :])  # (y-component, x-component)
        
        # Apply Poisson layer
        # poisson_outputs is a tuple of three tensors: (ux2, vy2, uyvx)
        poisson_outputs = self.poisson(vector_field)  # Each: [N, 1, H, W]
        
        # Concatenate original inputs with Poisson outputs
        # Original x: [N, 3, H, W]
        # poisson_outputs: [N, 1, H, W] each
        # Total channels after concatenation: 3 + 3 = 6
        concat = torch.cat([x, *poisson_outputs], dim=1)  # [N, 6, H, W]
        
        # Apply UnetDownBlock and ignore the downsampled output to maintain spatial dimensions
        x_before_pooling, _ = self.unet_down(concat)  # [N, n_base_features, H, W]
        
        # Apply ResNet block
        conv_res = self.resnet_block(x_before_pooling)  # [N, n_base_features, H, W]
        
        # Apply Final Conv2d to get pressure terms
        conv_out = self.final_conv(conv_res)  # [N, 1, H, W]
        
        return conv_out
