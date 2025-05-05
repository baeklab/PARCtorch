import sys

sys.path.append("../")

from PARCtorch.PARCv2 import PARCv2
import torch


def test_model(model=PARCv2()):
    assert model.differentiator is not None
    assert model.integrator is not None
    assert model.loss is not None


def test_PARCv2():
    model = PARCv2()

    # model initialization
    test_model(model)

    # check function
    assert model.check() == 1, f"ERROR: default manuscript configuration"

    model = PARCv2(loss=torch.nn.MSELoss)
    assert model.check() == 0, f"ERROR: warning system not working properly"
