import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d as TVDeformConv2d
from typing import Optional


class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers with optional deformable convolution.
    Includes a learnable offset for the deformable convolution.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding_mode="zeros", use_deform=False, deform_groups=1
    ):
        super(DoubleConv, self).__init__()
        self.use_deform = use_deform
        self.deform_groups = deform_groups

        # Offset convolution: predicts offsets for deformable convolution
        if use_deform:
            self.offset_conv = nn.Conv2d(
                in_channels,
                deform_groups * 2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            )
            self.conv1 = TVDeformConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                dilation=1,
                groups=1,
                bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
                bias=True
            )
        
        # Second convolution: always standard Conv2d
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            padding_mode=padding_mode,
            bias=True
        )
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        if self.use_deform:
            offset = self.offset_conv(x)
            x = self.conv1(x, offset)
        else:
            x = self.conv1(x)
        
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNetDownBlock(nn.Module):
    """
    U-Net Downsampling Block with optional deformable convolution.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding_mode="zeros", use_deform=False, deform_groups=1
    ):
        super(UNetDownBlock, self).__init__()
        self.doubleConv = DoubleConv(
            in_channels,
            out_channels,
            kernel_size,
            padding_mode,
            use_deform,
            deform_groups
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.doubleConv(x)
        return x


class UNetUpBlock(nn.Module):
    """
    U-Net Upsampling Block with optional deformable convolution.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        kernel_size=3,
        padding_mode="zeros",
        use_concat=True,
        use_deform=False,
        deform_groups=1
    ):
        super(UNetUpBlock, self).__init__()
        self.use_deform = use_deform
        self.use_concat = use_concat

        self.upConv = nn.Upsample(scale_factor=2, mode="bilinear")

        conv_in_channels = in_channels + skip_channels if use_concat else in_channels
        self.doubleConv = DoubleConv(
            conv_in_channels,
            out_channels,
            kernel_size,
            padding_mode,
            use_deform,
            deform_groups
        )

    def forward(self, x, skip_connection: Optional[torch.Tensor] = None):
        x = self.upConv(x)
        if self.use_concat and skip_connection is not None:
            x = torch.cat((skip_connection, x), dim=1)
        x = self.doubleConv(x)
        return x


class UNet(nn.Module):
    """
    U-Net Model with optional deformable convolutions.
    """
    def __init__(
        self,
        block_dimensions,
        input_channels,
        output_channels,
        kernel_size=3,
        padding_mode="zeros",
        up_block_use_concat=None,
        skip_connection_indices=None,
        use_deform=False,
        deform_groups=1
    ):
        super(UNet, self).__init__()
        self.use_deform = use_deform
        self.up_block_use_concat = up_block_use_concat
        self.skip_connection_indices = skip_connection_indices

        # Initial convolution
        self.doubleConv = DoubleConv(
            input_channels,
            block_dimensions[0],
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            use_deform=use_deform,
            deform_groups=deform_groups
        )

        # Downsampling path
        self.downBlocks = nn.ModuleList()
        in_channels = block_dimensions[0]
        skip_connection_channels = [in_channels]

        for out_channels in block_dimensions[1:]:
            down_block = UNetDownBlock(
                in_channels,
                out_channels,
                kernel_size,
                padding_mode,
                use_deform=use_deform,
                deform_groups=deform_groups
            )
            self.downBlocks.append(down_block)
            in_channels = out_channels
            skip_connection_channels.append(out_channels)

        # Upsampling path
        self.upBlocks = nn.ModuleList()
        num_up_blocks = len(block_dimensions) - 1

        if self.up_block_use_concat is None:
            self.up_block_use_concat = [True] * num_up_blocks
        else:
            assert (
                len(self.up_block_use_concat) == num_up_blocks
            ), "Length of up_block_use_concat must match number of up blocks."

        in_channels = block_dimensions[-1]
        skip_idx_counter = 0

        for idx in range(num_up_blocks):
            out_channels = (
                block_dimensions[-(idx + 2)] if idx != num_up_blocks - 1 else output_channels
            )
            use_concat = self.up_block_use_concat[idx]
            skip_channels = (
                skip_connection_channels[self.skip_connection_indices[skip_idx_counter]]
                if use_concat else 0
            )

            if use_concat:
                skip_idx_counter += 1

            up_block = UNetUpBlock(
                in_channels,
                out_channels,
                skip_channels,
                kernel_size,
                padding_mode,
                use_concat,
                use_deform=use_deform,
                deform_groups=deform_groups
            )
            self.upBlocks.append(up_block)
            in_channels = out_channels

        # Final convolution
        self.finalConv = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=1,
                padding=0,
                padding_mode=padding_mode
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=1,
                padding=0,
                padding_mode=padding_mode
            ),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = self.doubleConv(x)
        skip_connections = [x]

        # Downsampling path
        for down_block in self.downBlocks:
            x = down_block(x)
            skip_connections.append(x)

        # Upsampling path
        skip_idx_counter = 0
        for idx, up_block in enumerate(self.upBlocks):
            if self.up_block_use_concat[idx]:
                skip_connection = skip_connections[self.skip_connection_indices[skip_idx_counter]]
                skip_idx_counter += 1
            else:
                skip_connection = None
            x = up_block(x, skip_connection)

        x = self.finalConv(x)
        return x
