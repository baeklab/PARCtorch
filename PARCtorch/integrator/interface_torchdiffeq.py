from torchdiffeq import odeint, odeint_adjoint
import torch
from torch import nn


class Integrator(nn.Module):
    def __init__(self, use_adjoint: bool, odeint_kw: dict, **kwarg):
        super(Integrator, self).__init__(**kwarg)
        if use_adjoint:
            self.odeint_func = odeint_adjoint
        else:
            self.odeint_func = odeint
        self.odeint_kw = odeint_kw

    def forward(self, f, ic, t0, t1):
        all_time = torch.cat([t0.unsqueeze(0), t1])
        solution = self.odeint_func(f, ic, all_time, **self.odeint_kw)
        pred = solution[1:, ...]
        return pred
