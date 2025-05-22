import torch
import torch.nn as nn
from PARCtorch.integrator.poisson import PoissonBlock
from PARCtorch.utilities.load import get_device


class Integrator(nn.Module):
    """
    Constructor of Integrator

    Args
    clip (bool): whether to clip the state or velocity variable before each integration step
    list_poi_idx (list[tuple(int)]): List of channel indices for I/O of PoissonBlock.
                                     The last element in each tuple will be the channel index to output to, the 2nd and 3rd last will be assumed as a vector,
                                     and the rest as state variables.
    num_int (nn.Module): Numerical integrator
    list_dd_int (list[nn.Module]): List of data driven integrator. One per channel for state variables and one for the velocity as a whole.
    padding_mode (str): Padding mode
    finite_difference_method (nn.Module): Numercial method for finite difference calculation.
    poi_kernel_size (int, optional): Kernel size of the PoissonBlock. Default value is 3.
    n_poi_features (int, optional): Number of features in the PoissonBlock. Default value is 64.
    **kwarg: Other arguments that will be passed to nn.Module during initialization.
    """

    def __init__(
        self,
        clip: bool,
        list_poi_idx: list[tuple[int]],
        num_int: nn.Module,
        list_dd_int: list[nn.Module],
        padding_mode: str,
        finite_difference_method: nn.Module,
        poi_kernel_size: int = 3,
        n_poi_features: int = 64,
        device = None,
        **kwarg,
    ):
        super(Integrator, self).__init__(**kwarg)
        self.clip = clip
        self.numerical_integrator = num_int
        self.list_datadriven_integrator = nn.ModuleList(list_dd_int)
        self.list_poi_idx = list_poi_idx
        self.list_poi = nn.ModuleList()
        
        # Device selection
        if device is None:
            device = get_device()
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Initializing Poissons
        for each in list_poi_idx:
            self.list_poi.append(
                PoissonBlock(
                    len(each) - 1,
                    finite_difference_method,
                    poi_kernel_size,
                    n_poi_features,
                    padding_mode,
                )
            )

    def forward(self, f, ic, t0, t1):
        """
        Forward of Integrator. It will clip the current state and velocity variable (if necessary), go through the numerical integrator and then datadriven integrator.

        Args
        f (callable): Callable that returns time derivative
        ic (torch.tensor): 4-d tensor of Float with the shape (batch_size, channels, y, x), the initial condition to start integrating from
        t0 (float): Starting time
        t1 (torch.tensor): 1-d tensor of Float, the time points to integrate to


        Returns
        res (torch.tensor): 5-d tensor of Float with the shape (timesteps, batch_size, channels, y, x), the predicted state and velocity variables at each time in t1
        """
        all_time = torch.cat([t0.unsqueeze(0), t1])
        n_channel = ic.shape[1]
        n_state_var = n_channel - 2
        res = []
        current = ic
        for ts in range(1, all_time.shape[0]):
            if self.clip:
                current = torch.clamp(current, 0.0, 1.0)
            # Numerical integrator
            current, update = self.numerical_integrator(
                f, all_time[ts - 1], current, all_time[ts] - all_time[ts - 1]
            )
            # Poisson
            for i in range(len(self.list_poi)):
                idx_poi_in, idx_poi_out = (
                    self.list_poi_idx[i][:-1],
                    self.list_poi_idx[i][-1],
                )
                current[:, idx_poi_out : idx_poi_out + 1, :, :] = (
                    self.list_poi[
                        i
                    ](
                        torch.index_select(
                            current, 1, torch.tensor(idx_poi_in).to(self.device)
                        )
                    )
                )
            # Datadriven integrator
            current_ddi = []
            # State var first
            for i in range(n_state_var):
                if self.list_datadriven_integrator[i] is not None:
                    current_ddi.append(
                        self.list_datadriven_integrator[i](
                            update[:, i : i + 1, :, :],
                            current[:, i : i + 1, :, :],
                        )
                    )
                else:
                    current_ddi.append(current[:, i : i + 1, :, :])
            if self.list_datadriven_integrator[-1] is not None:
                current_ddi.append(
                    self.list_datadriven_integrator[-1](
                        update[:, -2:, :, :], current[:, -2:, :, :]
                    )
                )
            else:
                current_ddi.append(current[:, -2:, :, :])
            # Put them into an tensor
            current = torch.cat(current_ddi, 1)
            res.append(current.unsqueeze(0))
        res = torch.cat(res, 0)
        return res
