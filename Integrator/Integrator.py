import torch
import torch.nn as nn
from Poisson import Poisson


class Integrator(nn.Module):
    def __init__(self, clip, list_poi_idx, num_int, list_dd_int, padding_mode, finite_difference_method, **kwarg):
        super(Integrator, self).__init__(**kwarg)
        self.clip = clip
        self.numerical_integrator = num_int
        self.list_datadriven_integrator = list_dd_int
        self.list_poi_idx = self_poi_idx
        self.list_poi = []
        # Initializing Poissons
        for each in list_poi_idx:
            self.list_poi.append(Poisson(padding_mode, finite_difference_method))

    def forward(self, f, ic, t0, t1):
        '''
        Forward of Integrator. It will clip the current state and velocity variable (if necessary), go through the numerical integrator and then datadriven integrator.

        Parameters
        ----------
        f: function, the function that gives time derivative
        ic: 4-d tensor of Float with the shape (batch_size, channels, y, x), the initial condition to start integrating from
        t0: float, starting time
        t1: 1-d tensor of Float, the time points to calculate


        Returns
        -------
        res: 5-d tensor of Float with the shape (timesteps, batch_size, channels, y, x), the predicted state and velocity variables at each time in t1
        '''
        step_size, n_time_step = t1[0] - t0, t1.shape[0]
        n_channel = ic.shape[1]
        n_state_var = n_channel - 2
        res = []
        current = ic
        for _ in range(n_time_step):
            if self.clip:
                current = torch.clamp(current, 0.0, 1.0)
            # Numerical integrator
            current, update = self.numerical_integrator(f, current, step_size)
            # Poisson
            for i in range(len(list_poi)):
                idx_poi_in0, idx_poi_in1, idx_poi_out = list_poi_idx[i]
                current[:, idx_poi_out, :, :] = self.list_poi[i](torch.index_select(current, 1, torch.tensor([indx_poi_in0, indx_poi_in1])))
            # Datadriven integrator
            # State var first
            for i in range(n_state_var):
                if self.list_datadriven_integrator[i] is not None:
                    current[:, i, :, :] = self.list_datadriven_integrator[i](update[:, i:i+1, :, :], current[:, i:i+1, :, :])
            if self.list_datadriven_integrator[-1] is not None:
                current[:, -2:, :, :] = self.list_datadriven_integrator[-1](update[:, -2:, :, :], current[:, -2:, :, :])
            res.append(current.unsqueeze(0))
        res = torch.cat(res, 0)
        return res
