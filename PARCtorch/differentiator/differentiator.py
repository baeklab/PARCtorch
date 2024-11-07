import torch
import torch.nn as nn
from PARCtorch.differentiator.advection import Advection
from PARCtorch.differentiator.diffusion import Diffusion
from PARCtorch.differentiator.mappingandrecon import MappingAndRecon


class ADRDifferentiator(nn.Module):
    def __init__(
        self,
        n_state_var,
        n_fe_features,
        list_adv_idx,
        list_dif_idx,
        feature_extraction,
        padding_mode,
        finite_difference_method,
        spade_random_noise,
        **kwarg,
    ):
        """
        Constructor function of Differentiator

        Parameters
        ----------
        n_state_var: int, number of state varaibles
        n_fe_features: int, number of features from feature_extraction
        list_adv_idx: list of int, list of channel indices to calculate advection on
        list_dif_idx: list of int, list of channel indices to calculate diffusion on
        feature_extraction: torch.nn.Module, feature extraction network. It is expected to have input size of (batch_size, n_state_var+2, y, x) and output size of (batch_size, n_fe_features, y, x)
        padding_mode: string, padding mode for MappingAndRecon
        finite_difference_method: torch.nn.Module, numerical method to calculate finite difference. ```forward()``` of this module should be implemented to calculate spacial deriviatives.
        spade_random_noise (bool): whether to add noise in mapping and reconstruction modules or not.
        **kwarg: other arugments to be passed to torch.nn.Module

        Returns
        -------
        An instance of Differentiator
        """
        super(ADRDifferentiator, self).__init__(**kwarg)
        self.list_adv = nn.ModuleList()
        self.list_dif = nn.ModuleList()
        self.list_mar = nn.ModuleList()
        n_explicit_features = [0 for _ in range(n_state_var + 2)]
        self.feature_extraction = feature_extraction
        self.n_state_var = n_state_var
        # Initializing advections
        for i in range(n_state_var + 2):
            if i in list_adv_idx:
                self.list_adv.append(Advection(finite_difference_method))
                n_explicit_features[i] += 1
            else:
                self.list_adv.append(None)
        # Initializing diffusions
        for i in range(n_state_var + 2):
            if i in list_dif_idx:
                self.list_dif.append(Diffusion(finite_difference_method))
                n_explicit_features[i] += 1
            else:
                self.list_dif.append(None)
        # Initializing mapping and reconstruction
        # State variables first
        for i in range(n_state_var):
            if n_explicit_features[i] == 0:
                # No explicit features
                self.list_mar.append(None)
            else:
                # One or more explicit feature
                self.list_mar.append(
                    MappingAndRecon(
                        n_fe_features,
                        n_explicit_features[i],
                        1,
                        padding_mode,
                        spade_random_noise,
                    )
                )
        # Velocity variables second
        if n_explicit_features[-1] + n_explicit_features[-2] == 0:
            self.list_mar.append(None)
        else:
            self.list_mar.append(
                MappingAndRecon(
                    n_fe_features,
                    n_explicit_features[-1] + n_explicit_features[-2],
                    2,
                    padding_mode,
                    spade_random_noise,
                )
            )

    def forward(self, t, current):
        """
        Forward of differentiator. Advection and diffusion will be calculated per channel for those necessary and combined with dynamic features.
        Those that do not have explicit advection and diffusion calculation will have zero has output. This design choice was made because of
        certain integrtors (e.g. those in torchdiffeq) requires differentiator to have the same output and input shape.

        Parameters
        ----------
        t: torch.tensor, float scalar for current time
        current: 4-d tensor of Float with shape (batch_size, channels, y, x), the current state and velocity variables

        Returns
        -------
        t_dot: 4-d tensor of Float with the same shape as ```current```, the predicted time deriviatives on current state and velocity variables
        """
        dynamic_features = self.feature_extraction(current)
        t_dot = []
        # State variable
        for i in range(self.n_state_var):
            if self.list_mar[i] is not None:
                explicit_features = []
                if self.list_adv[i] is not None:
                    explicit_features.append(
                        self.list_adv[i](
                            current[:, i : i + 1, :, :], current[:, -2:, :, :]
                        )
                    )
                if self.list_dif[i] is not None:
                    explicit_features.append(
                        self.list_dif[i](current[:, i : i + 1, :, :])
                    )
                t_dot.append(
                    self.list_mar[i](
                        dynamic_features, torch.cat(explicit_features, 1)
                    )
                )
            else:
                t_dot.append(torch.zeros_like(current[:, i : i + 1, :, :]))
        # Velocity variable
        if self.list_mar[-1] is not None:
            explicit_features = []
            for i in [self.n_state_var, self.n_state_var + 1]:
                if self.list_adv[i] is not None:
                    explicit_features.append(
                        self.list_adv[i](
                            current[:, i : i + 1, :, :], current[:, -2:, :, :]
                        )
                    )
                if self.list_dif[i] is not None:
                    explicit_features.append(
                        self.list_dif[i](current[:, i : i + 1, :, :])
                    )
            t_dot.append(
                self.list_mar[-1](
                    dynamic_features, torch.cat(explicit_features, 1)
                )
            )
        else:
            t_dot.append(torch.zeros_like(current[:, -2:, :, :]))
        t_dot = torch.cat(t_dot, 1)
        return t_dot
