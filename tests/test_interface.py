import torch
from the_well.data.datasets import WellDataset
from PARCtorch.data.dataset import WellDatasetInterface


def test_well_dataset_interface_loading(patch_well_dataset_getitem, dummy_datapath):
    # Setup args expected by WellDataset
    well_dataset_args = {
        "path": str(dummy_datapath.parent)
    }

    # Create dataset interface
    dataset = WellDatasetInterface(
        well_dataset_args=well_dataset_args,
        future_steps=2,
        delta_t=0.5,
        add_constant_scalars=True,
    )

    # Basic checks
    assert len(dataset) == 2, "Expected 2 trajectories in dummy dataset"

    ic, t0, t1, gt = dataset[0]

    # Check time tensors
    assert isinstance(t0, torch.Tensor)
    assert isinstance(t1, torch.Tensor)
    assert t0.item() == 0.0
    assert torch.allclose(t1, torch.tensor([0.5, 1.0], dtype=torch.float32))

    # Check input condition shape
    assert ic.ndim == 3  # [C, H, W]
    C, H, W = ic.shape
    assert H == 32 and W == 32

    # Check ground truth 
    assert gt.ndim == 4  # [T, C, H, W]

    # Check constant scalar inclusion
    if dataset.add_constant_scalars:
        assert C > 1, "Expected constant scalars to increase channel count"
        assert torch.allclose(ic[0], ic[0][0, 0].expand(H, W)), "Constant scalar not broadcasted"

    # Check padding of missing fields (velocity_x and velocity_y)
    required_fields = {"velocity_x", "velocity_y"}
    field_names = {
        name for names in dataset.well_dataset.field_names.values() for name in names
    }
    missing = required_fields - field_names
    if missing:
        for f in missing:
            print(f"Missing field correctly padded: {f}")
        # Check zeros were added
        padded_channels = len(missing)
        assert C >= padded_channels, "Padded fields not added properly"
        assert (ic[-padded_channels:] == 0).all(), "Padded channels should be zeros"


def test_velocity_fields_are_loaded(patch_well_dataset_getitem, dummy_datapath):
    well_dataset_args = {
        "path": str(dummy_datapath.parent)
    }

    dataset = WellDatasetInterface(
        well_dataset_args=well_dataset_args,
        future_steps=1,
        delta_t=1.0,
        add_constant_scalars=False,
    )

    # Ensure velocity_x and velocity_y are not considered missing
    assert "velocity_x" not in dataset.missing_fields, "velocity_x should be present"
    assert "velocity_y" not in dataset.missing_fields, "velocity_y should be present"

    ic, _, _, gt = dataset[0]

    # Determine the number of channels before and after padding (there should be no change)
    # Since we did not add constant scalars, all channels come from input_fields
    input_channel_count = ic.shape[0]
    gt_channel_count = gt.shape[1]

    assert input_channel_count == 2, "input shape"
    assert gt_channel_count == 2, "input shape"

    # There should be no zero-padded channels at the end
    # Check the last few slices are not all zeros
    last_input_channel = ic[-1]
    last_gt_channel = gt[0, -1]

    assert not torch.allclose(last_input_channel, torch.zeros_like(last_input_channel)), \
        "Last input channel appears to be zero-padded (unexpected)"
    assert not torch.allclose(last_gt_channel, torch.zeros_like(last_gt_channel)), \
        "Last GT channel appears to be zero-padded (unexpected)"



