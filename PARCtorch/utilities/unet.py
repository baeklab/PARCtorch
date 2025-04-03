import torch
import torch.nn as nn
from typing import Optional


class UNetDownBlock(nn.Module):
    """
    U-Net Downsampling Block.

    Performs two convolutional operations followed by a max pooling operation
    to reduce the spatial dimensions of the input tensor while increasing the
    number of feature channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after convolution.
        kernel_size (int): Size of the convolutional kernels.
        normalization (nn.Module or None): Normalization layer.
        normalization_args (dict): Args for normalization layer.
        activation (nn.Module): Activation function.
        activation_args (dict): Args for activation function.
        padding_mode (str): Padding mode for convolutional layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        normalization,
        normalization_args,
        activation,
        activation_args,
        padding_mode,
    ):
        super(UNetDownBlock, self).__init__()
        self.padding_mode = padding_mode

        if normalization is None:
            doubleconv_modules = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
            ]
        else:
            doubleconv_modules = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                ),
                normalization(out_channels, **normalization_args),
                activation(**activation_args),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                normalization(out_channels, **normalization_args),
                activation(**activation_args),
            ]

        self.doubleConv = nn.Sequential(*doubleconv_modules)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.doubleConv(x)
        return x


class UNetUpBlock(nn.Module):
    """
    U-Net Upsampling Block.

    Performs upsampling using a transposed convolutional layer, optionally concatenates
    the corresponding skip connection from the downsampling path, and applies two
    convolutional operations to refine the features.

    Args:
        in_channels (int): Number of input channels from the previous layer.
        out_channels (int): Number of output channels after convolution.
        skip_channels (int): Number of channels from the skip connection.
        kernel_size (int): Size of the convolutional kernels.
        normalization (nn.Module or None): Normalization layer.
        normalization_args (dict): Args for normalization layer.
        activation (nn.Module): Activation function.
        activation_args (dict): Args for activation function.
        padding_mode (str): Padding mode for convolutional layers.
        use_concat (bool): Whether to concatenate skip connections.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size,
        normalization,
        normalization_args,
        activation,
        activation_args,
        padding_mode,
        use_concat,
    ):
        super(UNetUpBlock, self).__init__()
        self.padding_mode = padding_mode
        self.use_concat = use_concat

        self.upConv = nn.Upsample(scale_factor=2, mode="bilinear")

        # Calculate the number of input channels for the convolution after concatenation
        if use_concat:
            # If using skip connections, add the skip_channels to out_channels
            conv_in_channels = in_channels + skip_channels
        else:
            # If not using skip connections, the input channels remain the same
            conv_in_channels = in_channels

        # Define the double convolution layers with LeakyReLU activations
        if normalization is None:
            doubleconv_modules = [
                nn.Conv2d(
                    conv_in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
            ]
        else:
            doubleconv_modules = [
                nn.Conv2d(
                    conv_in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                ),
                normalization(out_channels, **normalization_args),
                activation(**activation_args),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                normalization(out_channels, **normalization_args),
                activation(**activation_args),
            ]

        self.doubleConv = nn.Sequential(*doubleconv_modules)

    def forward(self, x, skip_connection: Optional[torch.Tensor]):
        # Apply transposed convolution to upsample
        x = self.upConv(x)
        # Concatenate skip connection if enabled and available
        if self.use_concat and skip_connection is not None:
            # Concatenate along the channel dimension
            x = torch.cat((skip_connection, x), dim=1)
        # Apply double convolution
        x = self.doubleConv(x)
        return x


class UNet(nn.Module):
    """
    U-Net Model.

    Constructs a U-Net architecture with customizable depth and feature dimensions.
    Supports selective use of skip connections and concatenation in the upsampling path.

    Args:
        block_dimensions (list of int): List of feature dimensions for each block.
        output_channels (int): Number of output channels of the final layer.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        normalization (nn.Module or None, optional): Normalization layer. Default is None.
        normalization_args (dict, optional): Args for normalization layer. Default is an empty dictionary.
        activation (nn.Module, optional): Activation function. Default is nn.LeakyReLU.
        activation_args (dict, optional): Args for activation function. Default is {"negative_slope": 0.2}.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'zeros'.
        up_block_use_concat (list of bool, optional): Flags indicating whether to concatenate skip connections in each up block.
        skip_connection_indices (list of int, optional): Indices of skip connections to use in the upsampling path.
    """

    def __init__(
        self,
        block_dimensions,
        input_channels,
        output_channels,
        kernel_size=3,
        normalization=None,
        normalization_args={},
        activation=nn.LeakyReLU,
        activation_args={"negative_slope": 0.2},
        padding_mode="zeros",
        up_block_use_concat=None,
        skip_connection_indices=None,
    ):
        super(UNet, self).__init__()
        self.padding_mode = padding_mode

        # Store the concatenation flags and skip connection indices
        self.up_block_use_concat = up_block_use_concat
        self.skip_connection_indices = skip_connection_indices

        # Initial double convolution layer with LeakyReLU activations
        if normalization is None:
            doubleconv_modules = [
                nn.Conv2d(
                    input_channels,
                    block_dimensions[0],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
                nn.Conv2d(
                    block_dimensions[0],
                    block_dimensions[0],
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
            ]
        else:
            doubleconv_modules = [
                nn.Conv2d(
                    input_channels,
                    block_dimensions[0],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                ),
                normalization(block_dimensions[0], **normalization_args),
                activation(**activation_args),
                nn.Conv2d(
                    block_dimensions[0],
                    block_dimensions[0],
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                normalization(block_dimensions[0], **normalization_args),
                activation(**activation_args),
            ]

        self.doubleConv = nn.Sequential(*doubleconv_modules)

        # Downsampling path
        self.downBlocks = nn.ModuleList()  # List to hold downsampling blocks
        in_channels = block_dimensions[0]  # Initialize input channels
        skip_connection_channels = [
            in_channels
        ]  # List to store channels for skip connections

        # Construct the downsampling blocks
        for out_channels in block_dimensions[1:]:
            down_block = UNetDownBlock(
                in_channels,
                out_channels,
                kernel_size,
                normalization,
                normalization_args,
                activation,
                activation_args,
                padding_mode,
            )
            self.downBlocks.append(down_block)
            in_channels = out_channels
            skip_connection_channels.append(out_channels)

        # Upsampling path
        self.upBlocks = nn.ModuleList()
        num_up_blocks = len(block_dimensions) - 1

        # Ensure up_block_use_concat is provided and has the correct length
        if self.up_block_use_concat is None:
            self.up_block_use_concat = [True] * num_up_blocks
        else:
            assert (
                len(self.up_block_use_concat) == num_up_blocks
            ), "Length of up_block_use_concat must match the number of up blocks."

        # Initialize input channels for the first upsampling block
        in_channels = block_dimensions[-1]
        skip_idx_counter = 0

        # Construct the upsampling blocks
        for idx in range(num_up_blocks):
            if idx != num_up_blocks - 1:
                out_channels = block_dimensions[-(idx + 2)]
            else:
                out_channels = output_channels
            # Check if we should use concatenation in this block
            use_concat = self.up_block_use_concat[idx]
            if use_concat:
                skip_idx = self.skip_connection_indices[skip_idx_counter]
                skip_channels = skip_connection_channels[skip_idx]
                skip_idx_counter += 1
            else:
                skip_channels = 0

            # Create an upsampling block
            up_block = UNetUpBlock(
                in_channels,
                out_channels,
                skip_channels,
                kernel_size,
                normalization,
                normalization_args,
                activation,
                activation_args,
                padding_mode,
                use_concat,
            )
            self.upBlocks.append(up_block)
            # Update input channels for the next block
            in_channels = out_channels

        # Final convolution to map to the desired number of output channels
        if normalization is None:
            finalconv_modules = [
                nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
                nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                activation(**activation_args),
            ]
        else:
            finalconv_modules = [
                nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                normalization(output_channels, **normalization_args),
                activation(**activation_args),
                nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                normalization(output_channels, **normalization_args),
                activation(**activation_args),
            ]
        self.finalConv = nn.Sequential(*finalconv_modules)

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
                skip_idx = self.skip_connection_indices[skip_idx_counter]
                skip_connection = skip_connections[skip_idx]
                skip_idx_counter += 1
            else:
                skip_connection = None
            x = up_block(x, skip_connection)

        # Apply final convolution to get the desired output channels
        x = self.finalConv(x)
        return x
