import sys
sys.path.append("../")

from PARCtorch.PARCv2 import PARCv2
from PARCtorch.DPARC import DPARC 
from PARCtorch.model import PARC 

from differentiator.differentiator import ADRDifferentiator
from integrator.rk4 import RK4
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

def test_DPARC():
    model = DPARC()

    # model initialization
    test_model(model)

    # minimal structural check 
    assert issubclass(model, PARC)
    assert model.differentiator.feature_extractor.use_deform

    # default to manuscript - should be completed once manuscript is ready
