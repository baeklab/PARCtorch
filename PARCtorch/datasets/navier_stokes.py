from PARCtorch.data.dataset import GenericPhysicsDataset
from PARCtorch.utils.common import CACHE_DIR

import os

class NavierStokes(GenericPhysicsDataset):

    """
    TODO: Comment
    
    Args:
        split: Expected data split. Can be `train`, `test` TODO
        data_dir: Directory to read/write data. Defaults to None, in which case, the default CACHE_DIR is used.
        future_steps: Number of timesteps in the future the model will predict. Must be between 1 (single step prediction) and TODO (default).
        download: If False, the default, the data is assumed to be already in `data_dir`. If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    """

    url = "https://zenodo.org/records/13909869/files/NavierStokes.zip?download=1"

    def __init__(
        self,
        split=None,
        data_dir=None,
        future_steps=2,
        download = False,
    ):
        super().__init__()
        
        # Set up data directory
        if data_dir is None:
            root_dir = os.path.join(CACHE_DIR, "datasets")
            

        if download:
            self.download()
    
    def download(self):
        if self._check_exists():
            return
        
        download(url, './data/NavierStokes.zip')
        extract_zip('./data/NavierStokes.zip', './data/navier_stokes')


