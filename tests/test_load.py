import torch
from unittest.mock import patch
from PARCtorch.utilities.load import get_device, resolve_device

@patch("torch.cuda.is_available", return_value=True)
def test_get_device_cuda(mock_cuda):
    device = get_device()
    assert device.type == "cuda"

@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
def test_get_device_mps(mock_mps, mock_cuda):
    device = get_device()
    assert device.type == "mps"

@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
def test_get_device_cpu(mock_mps, mock_cuda):
    device = get_device()
    assert device.type == "cpu"

def test_resolve_device_string():
    device = resolve_device("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"

