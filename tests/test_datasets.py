from PARCtorch.data.dataset import WellDatasetInterface
from the_well.data import WellDataset
import os
import torch
import pytest

"""
# TODO: JC & XC to flesh out the test code below --> CW
# TODO: JC & XC to establish a quality guideline for hdf5 datasets --> BC
def testDataset():
    ds = parc.data.Dataset()
    assert hasattr(ds, 'fields')   # channel names, e.g. ['temperature', 'pressure'] 
    assert hasattr(ds, 'data')
    assert hasattr(ds, 'meta')
    assert hasattr(ds.meta, 'coefficients')


    assert ds.len() == 0
    # TODO: Check 'getItem'. Throw an error for ds.getItem(0)

def testBurgers():
    burgers = parc.Datasets.Burgers()
    assert os.path.exists('.cache/data/burgers')
    assert os.path.exists('.cache/data/burgers/XXXX.hdf5') # TODO: list of files
    assert type(burgers) == parc.data.Dataset
    # TODO: length of the dataset to be the same as the number of HDF5 files
    # TODO: Check number of snapshots

# TODO: write test functions for other datasets using the above Burgers example
#       as a template
"""


def is_same_shape(expected, actual):
    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a != b:
            return False
    return True

def check_directory():
    directory_path = "/standard/sds_baek_energetic/data/physics/the_well/datasets/"
    if os.path.exists(directory_path):
        return True
    else: 
        return False

@pytest.mark.skipif(not check_directory(), reason="Default WellDataset directory does not exist")
def test_dataset_thewell_trl2d():
    """
    Test WellDataset on turbulent_radiative_layer_2d
    """
    future_steps = 7
    ds = WellDatasetInterface(
        future_steps=future_steps,
        min_val=torch.tensor([0.03, 0.3217, 0.0541, 0.0, 0.0]),
        max_val=torch.tensor([3.16, 196.3853, 2.2342, 1.5395, 1.5395]),
        delta_t=0.01,
        add_constant_scalars=True,
        well_dataset_args={
            "well_base_path": "/standard/sds_baek_energetic/data/physics/the_well/datasets/",
            "well_dataset_name": "turbulent_radiative_layer_2D",
            "well_split_name": "train",
            "use_normalization": False,
        },
    )
    assert type(ds.well_dataset) is WellDataset
    assert torch.is_tensor(ds.min_val)
    assert torch.is_tensor(ds.max_val)
    assert type(ds.add_constant_scalars) is bool
    assert len(ds) == 6768
    for each in ds:
        ic, t0, t1, gt = each
        assert is_same_shape([5, 384, 128], ic.shape)  # t_cool, density, pressure, vx, vy
        assert is_same_shape([], t0.shape)
        assert is_same_shape([future_steps], t1.shape)
        assert is_same_shape([future_steps, 5, 384, 128], gt.shape)
        # First channel is the constant, t_cool
        assert (ic[0, :, :] == ic[0, 0, 0]).all()
        assert (gt[:, 0, :, :] == ic[0, 0, 0]).all()


@pytest.mark.skipif(not check_directory(), reason="Default WellDataset directory does not exist")
def test_dataset_thewell_trl2d_noconstant():
    """
    Test WellDataset on turbulent_radiative_layer_2d
    """
    future_steps = 7
    ds = WellDatasetInterface(
        future_steps=future_steps,
        min_val=torch.tensor([0.3217, 0.0541, 0.0, 0.0]),
        max_val=torch.tensor([196.3853, 2.2342, 1.5395, 1.5395]),
        delta_t=0.01,
        add_constant_scalars=False,
        well_dataset_args={
            "well_base_path": "/standard/sds_baek_energetic/data/physics/the_well/datasets/",
            "well_dataset_name": "turbulent_radiative_layer_2D",
            "well_split_name": "train",
            "use_normalization": False,
        },
    )
    assert type(ds.well_dataset) is WellDataset
    assert torch.is_tensor(ds.min_val)
    assert torch.is_tensor(ds.max_val)
    assert type(ds.add_constant_scalars) is bool
    assert len(ds) == 6768
    for each in ds:
        ic, t0, t1, gt = each
        assert is_same_shape([4, 384, 128], ic.shape)  # density, pressure, vx, vy
        assert is_same_shape([], t0.shape)
        assert is_same_shape([future_steps], t1.shape)
        assert is_same_shape([future_steps, 4, 384, 128], gt.shape)


@pytest.mark.skipif(not check_directory(), reason="Default WellDataset directory does not exist")
def test_dataset_thewell_gsrd():
    """
    Test WellDataset on gray_scott_reaction_diffusion
    """
    future_steps = 7
    ds = WellDatasetInterface(
        future_steps=future_steps,
        min_val=torch.tensor([0.0] * 6),
        max_val=torch.tensor([1.0] * 6),
        delta_t=1.0,
        add_constant_scalars=True,
        well_dataset_args={
            "well_base_path": "/standard/sds_baek_energetic/data/physics/the_well/datasets/",
            "well_dataset_name": "gray_scott_reaction_diffusion",
            "well_split_name": "train",
            "use_normalization": False,
        },
    )
    assert len(ds) == 954240
    for each in ds:
        ic, t0, t1, gt = each
        assert is_same_shape([6, 128, 128], ic.shape)  # f, k, A, B, vx==0, vy==0
        assert is_same_shape([], t0.shape)
        assert is_same_shape([future_steps], t1.shape)
        assert is_same_shape([future_steps, 6, 128, 128], gt.shape)
        # First and second channels are the constants,
        assert (ic[0, :, :] == ic[0, 0, 0]).all()
        assert (ic[1, :, :] == ic[1, 0, 0]).all()
        assert (gt[:, 1, :, :] == ic[1, 0, 0]).all()
        assert (gt[:, 0, :, :] == ic[0, 0, 0]).all()
        # This dataset does not have velocity, so the last 2 channels must be zero
        assert (ic[4:, :, :] == 0.0).all()
        assert (gt[:, 4:, :, :] == 0.0).all()


@pytest.mark.skipif(not check_directory(), reason="Default WellDataset directory does not exist")
def test_dataset_thewell_shear_flow():
    """
    Test WellDataset on shear flow
    """
    future_steps = 7
    ds = WellDatasetInterface(
        future_steps=future_steps,
        min_val=torch.tensor([0.0] * 6),
        max_val=torch.tensor([1.0] * 6),
        delta_t=1.0,
        add_constant_scalars=True,
        well_dataset_args={
            "well_base_path": "/standard/sds_baek_energetic/data/physics/the_well/datasets/",
            "well_dataset_name": "shear_flow",
            "well_split_name": "test",
            "use_normalization": False,
        },
    )
    
    for each in ds:
        ic, t0, t1, gt = each
        assert is_same_shape([6, 512, 256], ic.shape)  # Reynolds, Schmidt, tracer, pressure, velocity_x, velocity_y
        assert is_same_shape([], t0.shape)
        assert is_same_shape([future_steps], t1.shape)
        assert is_same_shape([future_steps, 6, 512, 256], gt.shape)
        # First and second channels are the constants,
        assert (ic[0, :, :] == ic[0, 0, 0]).all()
        assert (ic[1, :, :] == ic[1, 0, 0]).all()
        assert (gt[:, 1, :, :] == ic[1, 0, 0]).all()
        assert (gt[:, 0, :, :] == ic[0, 0, 0]).all()