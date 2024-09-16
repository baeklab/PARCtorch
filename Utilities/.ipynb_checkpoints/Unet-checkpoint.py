import torch
import torch.nn as nn

class UnetDownBlock(nn.Module):
    """
    U-Net Downsampling Block.

    Performs two convolutional operations followed by a max pooling operation
    to reduce the spatial dimensions of the input tensor while increasing the
    number of feature channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after convolution.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'zeros'.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding_mode='zeros'):
        super(UnetDownBlock, self).__init__()
        self.padding_mode = padding_mode

        # Define the double convolution layers with LeakyReLU activations
        self.doubleConv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=padding_mode),
            # Activation function
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Second convolutional layer
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=padding_mode),
            # Activation function
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Max pooling layer for downsampling (reduces spatial dimensions by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Print input shape for debugging
        print(f"DownBlock Input Shape: {x.shape}")
        # Apply double convolution
        x = self.doubleConv(x)
        print(f"After DoubleConv DownBlock: {x.shape}")
        # Apply max pooling for downsampling
        x_down = self.pool(x)
        print(f"After Pooling DownBlock: {x_down.shape}")
        # Return the output before pooling (for skip connections) and after pooling
        return x, x_down

class UnetUpBlock(nn.Module):
    """
    U-Net Upsampling Block.

    Performs upsampling using a transposed convolutional layer, optionally concatenates
    the corresponding skip connection from the downsampling path, and applies two
    convolutional operations to refine the features.

    Args:
        in_channels (int): Number of input channels from the previous layer.
        out_channels (int): Number of output channels after convolution.
        skip_channels (int, optional): Number of channels from the skip connection. Default is 0.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'zeros'.
        use_concat (bool, optional): Whether to concatenate skip connections. Default is True.
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, kernel_size=3, padding_mode='zeros', use_concat=True):
        super(UnetUpBlock, self).__init__()
        self.padding_mode = padding_mode
        self.use_concat = use_concat

        # Transposed convolution layer for upsampling (doubles spatial dimensions)
        self.upConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Calculate the number of input channels for the convolution after concatenation
        if use_concat:
            # If using skip connections, add the skip_channels to out_channels
            conv_in_channels = out_channels + skip_channels
        else:
            # If not using skip connections, the input channels remain the same
            conv_in_channels = out_channels

        # Define the double convolution layers with LeakyReLU activations
        self.doubleConv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=padding_mode),
            # Activation function
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Second convolutional layer
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=padding_mode),
            # Activation function
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x, skip_connection):
        # Print input shape for debugging
        print(f"UpBlock Input Shape: {x.shape}")
        # Apply transposed convolution to upsample
        x = self.upConv(x)
        print(f"After UpConv: {x.shape}")
        # Concatenate skip connection if enabled and available
        if self.use_concat and skip_connection is not None:
            print(f"Skip Connection Shape: {skip_connection.shape}")
            # Concatenate along the channel dimension
            x = torch.cat((skip_connection, x), dim=1)
            print(f"After Concatenation: {x.shape}")
        # Apply double convolution
        x = self.doubleConv(x)
        print(f"After DoubleConv UpBlock: {x.shape}")
        return x

class Unet(nn.Module):
    """
    U-Net Model.

    Constructs a U-Net architecture with customizable depth and feature dimensions.
    Supports selective use of skip connections and concatenation in the upsampling path.

    Args:
        block_dimensions (list of int): List of feature dimensions for each block.
        output_channels (int): Number of output channels of the final layer.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'zeros'.
        up_block_use_concat (list of bool, optional): Flags indicating whether to concatenate skip connections in each up block.
        skip_connection_indices (list of int, optional): Indices of skip connections to use in the upsampling path.
    """
    def __init__(self, block_dimensions, output_channels, kernel_size=3, padding_mode='zeros',
                 up_block_use_concat=None, skip_connection_indices=None):
        super(Unet, self).__init__()
        self.padding_mode = padding_mode

        # Store the concatenation flags and skip connection indices
        self.up_block_use_concat = up_block_use_concat
        self.skip_connection_indices = skip_connection_indices

        # Initial double convolution layer with LeakyReLU activations
        self.doubleConv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(block_dimensions[0], block_dimensions[0], kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=padding_mode),
            # Activation function
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Second convolutional layer
            nn.Conv2d(block_dimensions[0], block_dimensions[0], kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode=padding_mode),
            # Activation function
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Downsampling path
        self.downBlocks = nn.ModuleList()  # List to hold downsampling blocks
        in_channels = block_dimensions[0]  # Initialize input channels
        skip_connection_channels = []      # List to store channels for skip connections

        # Construct the downsampling blocks
        for out_channels in block_dimensions[1:]:
            # Create a downsampling block
            down_block = UnetDownBlock(in_channels, out_channels, kernel_size, padding_mode)
            self.downBlocks.append(down_block)
            # Update input channels for the next block
            in_channels = out_channels
            # Save the number of output channels for skip connections
            skip_connection_channels.append(out_channels)

        # Reverse the list to match the order of skip connections during upsampling
        skip_connection_channels = skip_connection_channels[::-1]

        # Upsampling path
        self.upBlocks = nn.ModuleList()  # List to hold upsampling blocks
        num_up_blocks = len(block_dimensions) - 1  # Number of upsampling blocks

        # Ensure up_block_use_concat is provided and has the correct length
        if self.up_block_use_concat is None:
            # If not provided, default to using concatenation in all up blocks
            self.up_block_use_concat = [True] * num_up_blocks
        else:
            # Assert that the length matches the number of up blocks
            assert len(self.up_block_use_concat) == num_up_blocks, \
                "Length of up_block_use_concat must match the number of up blocks."

        # Initialize input channels for the first upsampling block
        in_channels = block_dimensions[-1]
        skip_idx_counter = 0  # Counter for tracking skip_connection_indices

        # Construct the upsampling blocks
        for idx in range(num_up_blocks):
            # Determine the output channels for the current block
            out_channels = block_dimensions[-(idx + 2)]
            # Check if we should use concatenation in this block
            use_concat = self.up_block_use_concat[idx]

            if use_concat:
                # Get the index of the skip connection to use
                skip_idx = self.skip_connection_indices[skip_idx_counter]
                # Get the number of channels in the skip connection
                skip_channels = skip_connection_channels[skip_idx]
                # Increment the skip index counter
                skip_idx_counter += 1
            else:
                # If not using concatenation, skip channels are zero
                skip_channels = 0

            # Create an upsampling block
            up_block = UnetUpBlock(in_channels, out_channels, skip_channels, kernel_size, padding_mode, use_concat)
            self.upBlocks.append(up_block)
            # Update input channels for the next block
            in_channels = out_channels

        # Final convolution to map to the desired number of output channels
        self.finalConv = nn.Conv2d(block_dimensions[0], output_channels, kernel_size=1)

    def forward(self, x):
        # Print initial input shape for debugging
        print(f"Initial Input Shape: {x.shape}")
        # Apply initial double convolution
        x = self.doubleConv(x)
        print(f"After Initial DoubleConv: {x.shape}")
        skip_connections = []  # List to store outputs for skip connections

        # Downsampling path
        for down_block in self.downBlocks:
            # Apply downsampling block
            x, x_down = down_block(x)
            # Save the output before pooling for skip connections
            skip_connections.append(x)
            # Update x to be the downsampled output
            x = x_down

        # Reverse the skip connections to match the upsampling order
        skip_connections = skip_connections[::-1]

        # Upsampling path
        skip_idx_counter = 0  # Counter for tracking skip connections
        for idx, up_block in enumerate(self.upBlocks):
            if self.up_block_use_concat[idx]:
                # Get the index of the skip connection to use
                skip_idx = self.skip_connection_indices[skip_idx_counter]
                # Retrieve the corresponding skip connection
                skip_connection = skip_connections[skip_idx]
                # Increment the skip index counter
                skip_idx_counter += 1
            else:
                # If not using concatenation, set skip_connection to None
                skip_connection = None

            # Apply upsampling block with the skip connection
            x = up_block(x, skip_connection)

        # Apply final convolution to get the desired output channels
        x = self.finalConv(x)
        print(f"Final Output Shape: {x.shape}")
        return x
