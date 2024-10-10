# Differentiator/MappingAndRecon.py

import torch.nn as nn

# Import custom utilities
from PARCtorch.utilities.spade import SPADEGeneratorUnit
from PARCtorch.utilities.resnet import ResNet


class MappingAndRecon(nn.Module):
    def __init__(
        self,
        n_base_features=128,
        n_mask_channel=1,
        output_channel=1,
        padding_mode="zeros",
        add_noise=True,
    ):
        super(MappingAndRecon, self).__init__()
        self.add_noise = add_noise

        # Initialize SPADE generator unit
        self.spade = SPADEGeneratorUnit(
            in_channels=n_base_features,
            mask_channels=n_mask_channel,
            out_channels=n_base_features,
            padding_mode=padding_mode,
            kernel_size=1,
        )

        # Initialize ResNet block
        self.resnet = ResNet(
            n_base_features,
            [n_base_features, n_base_features],
            kernel_size=1,
            pooling=False,
            padding_mode=padding_mode,
        )
        # Final convolution layer
        self.conv_out = nn.Conv2d(
            in_channels=n_base_features,
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
        )

    def forward(self, dynamic_feature, advec_diff):
        """
        Forward pass of the MappingAndRecon.

        Args:
            dynamic_feature (torch.Tensor): Tensor of shape [N, C, H, W], dynamic features from the feature extraction network
            advec_diff (torch.Tensor): Tensor of shape [N, M, H, W], channel-concatenated advection and diffusion

        Returns:
            torch.Tensor: Output tensor of shape [N, output_channel, H, W]
        """
        spade_out = self.spade(dynamic_feature, advec_diff, self.add_noise)
        resnet_out = self.resnet(spade_out)
        conv_out = self.conv_out(resnet_out)

        return conv_out
