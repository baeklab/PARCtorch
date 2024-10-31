import torch.nn as nn


class NumericalIntegrator(nn.Module):
    def __init__(self, **kwarg):
        super(NumericalIntegrator, self).__init__()

    def forward(self, f, t, current, step_size):
        raise NotImplementedError
