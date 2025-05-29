import PARCtorch as parc


def test_dataset_navier_stokes():
    """
    Test built-in Navier-Stokes dataset
    """
    ds = parc.datasets.NavierStokes(path='dummy.py')