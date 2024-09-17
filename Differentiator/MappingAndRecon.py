import torch
import torch.nn as nn
from Utilities.Spade import SPADEGeneratorUnit
from Utilities.ResNet import ResNet  # Ensure the import path is correct based on your project structure

class MappingAndRecon(nn.Module):
    """
    MappingAndRecon Module

    This module integrates a SPADE (Spatially-Adaptive Denormalization) generator unit with a ResNet 
    architecture to perform mapping and reconstruction tasks. It processes dynamic features and 
    advection differences to produce the desired output.

    Args:
        n_base_features (int): Number of base feature channels for the SPADE and ResNet modules.
        n_mask_channel (int): Number of mask channels for the SPADE generator unit.
        output_channel (int): Number of output channels for the final convolution layer.
        padding_mode (str): Padding mode to be used in convolutional layers (e.g., 'reflect', 'constant').

    Attributes:
        spade (SPADEGeneratorUnit): Module for spatially-adaptive denormalization.
        resnet (ResNet): Residual network module for feature mapping and reconstruction.
        conv_out (nn.Conv2d): Final convolutional layer to produce the desired output channels.
    """
    def __init__(self, n_base_features, n_mask_channel, output_channel, padding_mode):
        super(MappingAndRecon, self).__init__()
        
        # Initialize the SPADE generator unit
        self.spade = SPADEGeneratorUnit(
            in_channels=n_base_features,   # Number of input channels for SPADE (dynamic_feature)
            mask_channels=n_mask_channel,  # Number of mask channels (advec_diff)
            out_channels=n_base_features,  # Number of output channels from SPADE
            padding_mode=padding_mode      # Padding mode as specified
        )

        # Initialize the ResNet module
        # - in_channels: Number of input channels to ResNet (output from SPADE)
        # - block_dimensions: List defining the number of feature channels for each ResNet block
        #   Here, [n_base_features, n_base_features] creates two ResNet blocks with the same number of channels
        # - kernel_size: Size of the convolutional kernels within ResNet blocks
        # - pooling: Whether to apply max pooling after each ResNet block (False to maintain spatial dimensions)
        # - padding_mode: Padding mode as specified
        self.resnet = ResNet(
            in_channels=n_base_features,
            block_dimensions=[n_base_features] * 2,  # Example: [128, 128] if n_base_features=128
            kernel_size=3,                            # Using a 3x3 kernel for better spatial feature extraction
            pooling=False,                            # Disable pooling to maintain spatial dimensions
            padding_mode=padding_mode
        )

        # Initialize the final convolutional layer
        # - in_channels: Number of input channels (output from ResNet)
        # - out_channels: Desired number of output channels
        # - kernel_size: 1x1 convolution to adjust channel dimensions without altering spatial dimensions
        # - padding: 0 to apply 'valid' padding (no padding), maintaining spatial dimensions
        self.conv_out = nn.Conv2d(
            in_channels=n_base_features,  # Must match the output channels from ResNet
            out_channels=output_channel,  # Desired output channels (e.g., for segmentation, masks, etc.)
            kernel_size=1,                # 1x1 convolution for channel adjustment
            padding=0                      # No padding to preserve spatial dimensions
        )

    def forward(self, dynamic_feature, advec_diff):
        """
        Forward pass of the MappingAndRecon module.

        Args:
            dynamic_feature (torch.Tensor): Tensor containing dynamic features with shape [N, C, H, W].
            advec_diff (torch.Tensor): Tensor containing advection differences with shape [N, M, H, W].

        Returns:
            torch.Tensor: Output tensor after processing through SPADE, ResNet, and final convolution,
                          with shape [N, output_channel, H, W].
        """
        # Pass the dynamic features and advection differences through the SPADE generator unit
        # This step applies spatially-adaptive normalization based on the mask provided by advec_diff
        x = self.spade(dynamic_feature, advec_diff)
        
        # Pass the output from SPADE through the ResNet module
        # ResNet processes the features through residual blocks for deeper feature extraction
        x = self.resnet(x)
        
        # Apply the final 1x1 convolution to adjust the number of output channels
        # This layer transforms the feature maps to the desired output format
        x = self.conv_out(x)
        
        # Return the final output
        return x
