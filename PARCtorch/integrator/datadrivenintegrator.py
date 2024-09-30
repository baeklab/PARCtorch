import torch.nn as nn
import torch.nn.functional as F
from PARCtorch.utilities.spade import SPADEGeneratorUnit
from PARCtorch.utilities.resnet import ResNet


class DataDrivenIntegrator(nn.Module):
    def __init__(
        self,
        n_io: int,
        n_base_features: int,
        kernel_size: int = 1,
        padding_mode: str = "constant",
        spade_random_noise: bool = True,
        **kwarg,
    ):
        """
        Args
        n_io (int): number of channels for input and output
        n_base_features (int): number of channels for all hideen layers
        kernel_size (int, optional): kernel size for all convolution layers, except SPADE. Default value is 1.
        padding_mode (str, optional): padding mode. Default value is constant.
        """
        super(DataDrivenIntegrator, self).__init__(**kwarg)
        self.padding_mode = padding_mode
        self.pad_size = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(n_io, n_base_features, kernel_size)
        self.conv2 = nn.Conv2d(n_base_features, n_base_features, kernel_size)
        self.spade = SPADEGeneratorUnit(
            n_base_features,
            n_base_features,
            n_io,
            kernel_size,
            3,
            padding_mode,
        )
        self.resnet = ResNet(
            n_base_features,
            [n_base_features, n_base_features],
            kernel_size,
            False,
            padding_mode,
        )
        self.conv3 = nn.Conv2d(n_base_features, n_io, kernel_size)
        self.spade_random_noise = spade_random_noise

    def forward(self, update, current):
        """
        Args
        update (torch.tensor): The change reported by the numerical integrator. Number of channels must equal to ```n_io```.
        current (torch.tensor): The state and/or velocity variables at current time step. Number of channels must equal to ```n_io```.

        Returns
        out (torch.tensor): The predicted state and/or velocity variables at the next time step. Shape is the same as ```update``` and ```current```
        """
        out = F.pad(
            update,
            (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            self.padding_mode,
        )
        out = self.conv1(out)
        out = F.relu(out)
        out = F.pad(
            out,
            (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            self.padding_mode,
        )
        out = self.conv2(out)
        out = F.relu(out)
        out = self.spade(out, current, self.spade_random_noise)
        out = self.resnet(out)
        out = self.conv3(out)
        return out
