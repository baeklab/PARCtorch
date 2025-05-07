import PARCtorch as parc
import os

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


