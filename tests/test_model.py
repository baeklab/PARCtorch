from PARCtorch.PARCv2 import PARCv2
from PARCtorch.differentiator.differentiator import ADRDifferentiator
from PARCtorch.integrator.rk4 import RK4
import torch

def test_model(model=PARCv2()):
    assert model.differentiator is not None
    assert model.integrator is not None
    assert model.loss is not None


def test_PARCv2():
    """ boiler-plate for minimal checking for variant of PARC follows the manuscript """ 
    model = PARCv2()

    # model initialization
    test_model(model)

    # default to manuscript
    assert isinstance(model.differentiator, ADRDifferentiator)
    assert isinstance(model.integrator.numerical_integrator, RK4)
    assert isinstance(model.loss, torch.nn.L1Loss)
