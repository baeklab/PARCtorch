# Differentiator/MappingAndRecon.py

import torch
import torch.nn as nn
import numpy as np

# Import custom utilities
from Utilities.Spade import SPADEGeneratorUnit
from Utilities.ResNet import ResNet  # Ensure ResNet is correctly implemented

class MappingAndRecon(nn.Module):
    def __init__(self, n_base_features=128, n_mask_channel=1, 
                 output_channel=1, padding_mode="constant"):
        super(MappingAndRecon, self).__init__()
        
        # Initialize SPADE generator unit
        self.spade = SPADEGeneratorUnit(
            in_channels=n_base_features,
            mask_channels=n_mask_channel,
            out_channels=n_base_features,
            upsampling=False,
            padding_mode=padding_mode  # Corrected parameter name and casing
        )
        
        # Initialize ResNet block
        self.resnet = ResNet(
            in_channels=n_base_features,
            feat_dim=n_base_features,       # feat_dim corresponds to out_channels
            kernel_size=1,
            reps=2,
            pooling=False,
            padding_mode=padding_mode        # Corrected parameter name and casing
        )
        
        # Final convolution layer
        self.conv_out = nn.Conv2d(
            in_channels=n_base_features,
            out_channels=output_channel,
            kernel_size=1,
            padding=0  # 'valid' padding in Keras corresponds to padding=0 in PyTorch
        )
    
    def forward(self, dynamic_feature, advec_diff):
        """
        Forward pass of the MappingAndRecon.
        
        Args:
            dynamic_feature (torch.Tensor): Tensor of shape [N, C, H, W] where C = n_base_features
            advec_diff (torch.Tensor): Tensor of shape [N, M, H, W] where M = n_mask_channel
        
        Returns:
            torch.Tensor: Output tensor of shape [N, output_channel, H_out, W_out]
        """
        # Pass through SPADE generator unit
        spade_out = self.spade(dynamic_feature, advec_diff)
        
        # Pass through ResNet block
        resnet_out = self.resnet(spade_out)
        
        # Final convolution
        conv_out = self.conv_out(resnet_out)
        
        return conv_out
