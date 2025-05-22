from PARCtorch.LatentPARC import LatentPARC
from PARCtorch.model import PARC
from PARCtorch.utilities.autoencoder import Encoder, Decoder
import torch


def test_latentparc_default():
    # This test requires a GPU to run
    model = LatentPARC().cuda()
    assert issubclass(model, PARC)
    assert isinstance(model.encoder, Encoder)
    assert isinstance(model.decoder, Decoder)
    assert issubclass(model.differentiator, torch.nn.Module)
    assert issubclass(model.integrator, torch.nn.Module)
    # freeze_encoder_decoder functionalities
    model.freeze_encoder_decoder()
    for p in model.encoder.paramters():
        assert p.reqires_grad is False
    for p in model.decoder.paramters():
        assert p.requires_grad is False
    # Forward pass
    # TODO: complete upon submission of draft to avoid leaking implementation details
    ic = torch.randn(4, 5, 128, 256, dtype=torch.float32, device="cuda")
    t0 = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    t1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
    gt = torch.randn(3, 4, 5, 128, 256, dtype=torch.float32, device="cuda")
    # Backward pass
    # TODO: complete upon submission of draft to avoid leaking implementation details
