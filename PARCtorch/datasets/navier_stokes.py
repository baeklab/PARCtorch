from PARCtorch.data.dataset import GenericPhysicsDataset
from PARCtorch.utils.common import CACHE_DIR

from PARCtorch.datasets.utils import download, extract_zip

import os


class NavierStokes(GenericPhysicsDataset):

    """
    Navier-Stokes data set. TODO: More details
    
    Args:
        split: Expected data split. Can be `train`, `test` TODO
        data_dir: Directory to read/write data. Defaults to None, in which case,
                  data will be stored in `CACHE_DIR/datasets/NavierStokes`.
                  If the data does not exist in the specified directory,
                  it will be automatically downloaded.
        future_steps: Number of timesteps in the future the model will predict.
                      Must be between 1 (single step prediction) and TODO (default).
    """

    url = "https://zenodo.org/records/13909869/files/NavierStokes.zip?download=1"

    def __init__(
        self, split=None, data_dir=None, future_steps=2,
    ):
        super().__init__()

        # Set up data directory
        if data_dir is None:
            data_dir = os.path.join(CACHE_DIR, 'datasets', 'NavierStokes')
        self.data_dir = data_dir
        self.zip_dir = os.path.join(self.data_dir, 'NavierStokes.zip')

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Download and unzip data if it doesn't exist already.
        self.download(force=False)

        from pathlib import Path
        filelist = Path(self.data_dir).glob('*.*')


    def download(self, force=False):
        if not os.path.exists(self.zip_dir):
            download(self.url, self.zip_dir)
            extract_zip(self.zip_dir, self.data_dir)
