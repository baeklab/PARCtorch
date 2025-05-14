import torch
import torch.nn as nn

from PARCtorch.utilities.unet import UNet
from PARCtorch.differentiator.finitedifference import FiniteDifference
from differentiator.differentiator import ADRDifferentiator
from integrator.rk4 import RK4
from integrator.integrator import Integrator
from model import PARC


class DPARC(PARC):
    def __init__(self, differentiator=None, integrator=None, loss=None, **kwargs):
        """
        Constructor of DPARC, a deformable convolution version of PARCv2.
        """
        n_fe_features = 128
        right_diff = FiniteDifference(padding_mode="replicate").cuda()

        if differentiator is None:
            unet_em = UNet(
                [64, 64 * 2, 64 * 4, 64 * 8, 64 * 16],
                5,
                n_fe_features,
                up_block_use_concat=[False, True, False, True],
                skip_connection_indices=[2, 0],
                use_deform=True,
                deform_groups=1,
            )

            differentiator = ADRDifferentiator(
                3,  # 3 state variables (T, p, mu)
                n_fe_features,
                [0, 1, 2, 3, 4],  # all channels for advection
                [0],              # T for diffusion
                unet_em,
                "constant",
                right_diff,
                spade_random_noise=False,
            ).cuda()

        if integrator is None:
            rk4_int = RK4().cuda()
            ddi_list = [None] * 5  # one for each state var + 2 velocity
            integrator = Integrator(
                True,
                [],
                rk4_int,
                ddi_list,
                "constant",
                right_diff,
            ).cuda()

        if loss is None:
            loss = nn.L1Loss().cuda()

        super(DPARC, self).__init__(differentiator, integrator, loss, **kwargs)

        print(type(self.differentiator), type(self.integrator), type(self.loss))

    def check(self):
        diff = isinstance(self.differentiator, ADRDifferentiator)
        intg = isinstance(self.integrator.numerical_integrator, RK4)
        loss = isinstance(self.loss, torch.nn.L1Loss)
        return diff and intg and loss
