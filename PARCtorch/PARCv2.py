import torch

from PARCtorch.models.unet import UNet
from PARCtorch.differentiator.finitedifference import FiniteDifference
from differentiator.differentiator import ADRDifferentiator
from integrator.rk4 import RK4
from integrator.integrator import Integrator
from model import PARC


class PARCv2(PARC):
    def __init__(self, differentiator=None, integrator=None, loss=None, **kwargs):
        """
        Constructor of PARCv2.

        Parameters
        ----------
        differentiator: PARCv2.Differentiator.Differentiator, the differentiator
        integrator: PARCv2.Integrator.Integrator, the numerical and data-driven (if necessary) integrator
        loss: function, loss function, typicially would be torch.nn.MSELoss or torch.nn.L1Loss
        **kwargs: other parameters that will be passed onto torch.nn.Module

        Returns
        -------
        An instance of PARCv2
        """
        right_diff = FiniteDifference(padding_mode="replicate").cuda()
        if differentiator is None:
            unet = UNet(
                [64, 64 * 2, 64 * 4, 64 * 8, 64 * 16],
                5,
                128,
                up_block_use_concat=[False, True, False, True],
                skip_connection_indices=[2, 0],
            )
            differentiator = ADRDifferentiator(
                3,
                128,
                [0, 1, 2, 3, 4],
                [0],
                unet,
                "constant",
                right_diff,
                spade_random_noise=True,
            ).cuda()

        if integrator is None:
            rk4 = RK4().cuda()
            integrator = Integrator(
                True, [], rk4, [None] * 5, "constant", right_diff
            ).cuda()

        if loss is None:
            loss = torch.nn.L1Loss().cuda()
        super(PARCv2, self).__init__(differentiator, integrator, loss, **kwargs)
