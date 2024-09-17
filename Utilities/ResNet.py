# Utilities/ResNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ResNetBlock(nn.Module):
    """
    Residual Block for ResNet with support for changing feature dimensions.

    This block performs two convolutional operations, each followed by ReLU activation,
    and adds the input tensor to the output of the second convolution (skip connection).
    If the number of input and output channels differ, a 1x1 convolution is applied to
    the input to match the dimensions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'reflect'.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding_mode: str = 'reflect'):
        super(ResNetBlock, self).__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Determine padding size to maintain spatial dimensions
        self.pad_size = (kernel_size - 1) // 2

        # First convolutional layer followed by ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer followed by ReLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Skip connection: 1x1 convolution if in_channels != out_channels
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply padding before the first convolution
        out = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode=self.padding_mode)
        out = self.conv1(out)
        out = self.relu1(out)

        # Apply padding before the second convolution
        out = F.pad(out, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode=self.padding_mode)
        out = self.conv2(out)

        # Adjust input dimensions if necessary
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
        else:
            skip = x

        # Add skip connection and apply ReLU
        out += skip
        out = self.relu2(out)

        return out

class ResNet(nn.Module):
    """
    ResNet model consisting of multiple residual blocks with varying feature dimensions.

    The model initializes with a convolutional layer and iteratively adds residual blocks
    based on the provided `block_dimensions`. Optionally, it applies max pooling after
    each residual block to reduce spatial dimensions.

    Args:
        in_channels (int): Number of input channels.
        block_dimensions (List[int]): List specifying the number of feature channels for each residual block.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        pooling (bool, optional): Whether to apply max pooling after each residual block. Default is True.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'reflect'.
    """
    def __init__(
        self, 
        in_channels: int, 
        block_dimensions: List[int], 
        kernel_size: int = 3, 
        pooling: bool = True, 
        padding_mode: str = 'reflect'
    ):
        super(ResNet, self).__init__()
        self.padding_mode = padding_mode
        self.pooling = pooling
        self.kernel_size = kernel_size

        # Initialize the first convolutional layer (without padding)
        self.initial_conv = nn.Conv2d(in_channels, block_dimensions[0], kernel_size=kernel_size, padding=0)
        self.initial_relu = nn.ReLU(inplace=True)

        # Create residual blocks based on block_dimensions
        self.resnet_units = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i in range(len(block_dimensions)):
            if i == 0:
                # First block uses the initial_conv's output channels as in_channels
                res_block = ResNetBlock(
                    in_channels=block_dimensions[0],
                    out_channels=block_dimensions[0],
                    kernel_size=kernel_size,
                    padding_mode=padding_mode
                )
            else:
                # Subsequent blocks may change the number of channels
                res_block = ResNetBlock(
                    in_channels=block_dimensions[i-1],
                    out_channels=block_dimensions[i],
                    kernel_size=kernel_size,
                    padding_mode=padding_mode
                )
            self.resnet_units.append(res_block)

            # Optionally add a pooling layer after each residual block except the last one
            if self.pooling and i < len(block_dimensions) - 1:
                self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.pool_layers.append(None)  # Placeholder for consistency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial padding and convolution
        out = F.pad(
            x, 
            (
                (self.kernel_size - 1) // 2, 
                (self.kernel_size - 1) // 2,
                (self.kernel_size - 1) // 2, 
                (self.kernel_size - 1) // 2
            ),
            mode=self.padding_mode
        )
        out = self.initial_conv(out)
        out = self.initial_relu(out)
        print(f"After Initial Conv: {out.shape}")

        # Apply residual blocks with optional pooling
        for idx, res_block in enumerate(self.resnet_units):
            out = res_block(out)
            print(f"After ResNetBlock {idx + 1}: {out.shape}")

            # Apply pooling if defined
            if self.pool_layers[idx] is not None:
                out = self.pool_layers[idx](out)
                print(f"After Pooling {idx + 1}: {out.shape}")

        return out
