import torch.nn as nn


class PARCv2(nn.Module):
    def __init__(self, differentiator, integrator, loss, **kwargs):
        '''
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
        '''
        super(PARCv2, self).__init__(**kwargs)
        
        self.differentiator = differentiator
        self.integrator = integrator
        self.loss = loss

    def forward(self, ic, t0, t1):
        '''
        Forward of PARCv2. Essentially a call to the integrator with the differentiator.

        Parameters:
        -----------
        ic: 4-d tensor of Float with the shape of (batch_size, channels, y, x), initial condition.
        t0: float, starting time of the initial condition
        t1: 1-d tensor of Float, time point that PARCv2 will return forward pass on

        Returns
        -------
        res: 5-d tnsor of Float with the shape of (ts, batch_size, channels, y, x), predicted sequences at each time point in t1
             ts is the length of t1
        '''
        return self.integrator(self.differentiator, ic, t0, t1)
