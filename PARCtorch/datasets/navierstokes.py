from PARCtorch.data.dataset import GenericPhysicsDataset

class NavierStokes(GenericPhysicsDataset):
    """
    TODO: Comment
    Inspired by: https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html#MNIST
    """

    url = "https://zenodo.org/records/13909869/files/NavierStokes.zip?download=1"

    def __init__(
        self,
        download = False,
    ):
        super().__init__()
        # TODO: Implement initialization

        if download:
            self.download()
    
    def download(self):
        if self._check_exists():
            return
        
        download(url, './data/NavierStokes.zip')
        extract_zip('./data/NavierStokes.zip', './data/navier_stokes')


