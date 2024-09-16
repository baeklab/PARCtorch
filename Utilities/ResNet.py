import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """
    Residual block for ResNet.

    This block performs two convolutional operations, each followed by ReLU activation, and 
    adds the input tensor to the output of the second convolution (skip connection).

    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Size of the convolutional kernel.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'constant' (zero padding).
    """
    def __init__(self, in_channels, kernel_size=3, padding_mode='constant'):
        super(ResNetBlock, self).__init__()
        self.padding_mode = padding_mode

        # Padding size for 'valid' convolutions (to match the TensorFlow implementation)
        self.pad_size = (kernel_size - 1) // 2

        # First convolutional layer followed by ReLU
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               padding=0)  # no padding in the Conv2d, we pad externally
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer followed by ReLU
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               padding=0)  # no padding in the Conv2d, we pad externally
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply padding before the first convolution
        x_padded = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                         mode=self.padding_mode)
        # First convolution + ReLU
        out = self.conv1(x_padded)
        out = self.relu1(out)

        # Apply padding before the second convolution
        out_padded = F.pad(out, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                           mode=self.padding_mode)
        # Second convolution
        out = self.conv2(out_padded)

        # Add input (skip connection) and apply ReLU
        out += x
        out = self.relu2(out)

        return out

class ResNet(nn.Module):
    """
    ResNet model consisting of multiple residual blocks and optional pooling.

    The model applies two initial convolutional layers followed by 'reps' number
    of residual blocks. Optionally applies max pooling at the end.

    Args:
        in_channels (int): Number of input channels.
        feat_dim (int): Number of output feature channels.
        kernel_size (int): Size of the convolutional kernel.
        reps (int): Number of residual units.
        pooling (bool, optional): Whether to apply max pooling at the end. Default is True.
        padding_mode (str, optional): Padding mode for convolutional layers. Default is 'constant'.
    """
    def __init__(self, in_channels, feat_dim, kernel_size=3, reps=2, pooling=True, padding_mode='constant'):
        super(ResNet, self).__init__()
        self.padding_mode = padding_mode
        self.feat_dim = feat_dim
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.reps = reps

        # Padding size for 'valid' convolutions
        self.pad_size = (kernel_size - 1) // 2

        # Initial convolutional layers followed by ReLU
        self.conv1 = nn.Conv2d(in_channels, feat_dim, kernel_size=kernel_size,
                               padding=0)  # no padding in the Conv2d, we pad externally
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=kernel_size,
                               padding=0)  # no padding in the Conv2d, we pad externally
        self.relu2 = nn.ReLU(inplace=True)

        # Residual units
        self.resnet_units = nn.ModuleList([
            ResNetBlock(feat_dim, kernel_size=kernel_size, padding_mode=padding_mode) for _ in range(reps)
        ])

        # Max pooling layer
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

    def forward(self, x):
        # Apply padding before the first convolution
        x_padded = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                         mode=self.padding_mode)
        # First convolution + ReLU
        out = self.conv1(x_padded)
        out = self.relu1(out)

        # Apply padding before the second convolution
        out_padded = F.pad(out, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                           mode=self.padding_mode)
        # Second convolution + ReLU
        out = self.conv2(out_padded)
        out = self.relu2(out)

        # Apply the residual units
        for unit in self.resnet_units:
            out = unit(out)

        # Apply max pooling if enabled
        if self.pool is not None:
            out = self.pool(out)

        return out
