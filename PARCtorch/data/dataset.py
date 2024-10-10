import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from tqdm import tqdm


def validate_data_format(
    data_dirs, future_steps=1, min_max_path=None, required_channels=None
):
    """
    Validates the format of the data directories to ensure they contain properly formatted .npy files
    and corresponding min_max.json files.

    Args:
        data_dirs (list of str): List of directories containing preprocessed `.npy` files.
        future_steps (int): Number of timesteps in the future the model will predict.
        min_max_path (str, optional): Path to the JSON file containing min and max values for each channel.
                                      If None, it will look for 'min_max.json' in each data directory.
        required_channels (int, optional): Number of channels expected in the data.

    Raises:
        ValueError: If any of the validation checks fail.
    """
    logging.info("Starting data validation...")

    all_files = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            raise ValueError(
                f"Data directory '{data_dir}' does not exist or is not a directory."
            )
        dir_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npy")
            ]
        )
        if not dir_files:
            logging.warning(f"No .npy files found in directory '{data_dir}'.")
        all_files.extend(dir_files)

    if not all_files:
        raise ValueError(
            "No .npy files found in any of the specified directories."
        )

    # Load min and max values
    if min_max_path is None:
        min_max_files = [
            os.path.join(data_dir, "min_max.json") for data_dir in data_dirs
        ]
        channel_min = []
        channel_max = []
        for mm_path in min_max_files:
            if not os.path.exists(mm_path):
                raise FileNotFoundError(
                    f"Min and max values file not found at '{mm_path}'. "
                    "Please ensure the file exists."
                )
            with open(mm_path, "r") as f:
                min_max = json.load(f)
            if "channel_min" not in min_max or "channel_max" not in min_max:
                raise ValueError(
                    f"'channel_min' or 'channel_max' not found in '{mm_path}'."
                )
            channel_min.extend(min_max["channel_min"])
            channel_max.extend(min_max["channel_max"])
    else:
        if not os.path.exists(min_max_path):
            raise FileNotFoundError(
                f"Min and max values file not found at '{min_max_path}'. "
                "Please ensure the file exists."
            )
        with open(min_max_path, "r") as f:
            min_max = json.load(f)
        if "channel_min" not in min_max or "channel_max" not in min_max:
            raise ValueError(
                f"'channel_min' or 'channel_max' not found in '{min_max_path}'."
            )
        channel_min = min_max["channel_min"]
        channel_max = min_max["channel_max"]

    num_channels = len(channel_min)
    if len(channel_max) != num_channels:
        raise ValueError(
            "Length of 'channel_min' and 'channel_max' must be the same."
        )

    if required_channels is not None:
        if num_channels != required_channels:
            raise ValueError(
                f"Number of channels in min_max.json ({num_channels}) does not match "
                f"the required_channels ({required_channels})."
            )
        logging.info(f"Number of channels validated: {num_channels}")

    # Validate each .npy file
    logging.info("Validating .npy files...")
    for file in tqdm(all_files, desc="Validating files"):
        try:
            data = np.load(file, mmap_mode="r")
        except Exception as e:
            raise ValueError(f"Error loading file '{file}': {e}")

        if data.ndim != 4:
            raise ValueError(
                f"File '{file}' has {data.ndim} dimensions; expected 4 dimensions (timesteps, channels, height, width)."
            )

        timesteps, channels, height, width = data.shape
        if required_channels is not None and channels != required_channels:
            raise ValueError(
                f"File '{file}' has {channels} channels; expected {required_channels} channels."
            )
        elif required_channels is None and channels != num_channels:
            raise ValueError(
                f"File '{file}' has {channels} channels; expected {num_channels} channels based on min_max.json."
            )

        if timesteps < future_steps + 1:
            raise ValueError(
                f"File '{file}' has {timesteps} timesteps; requires at least {future_steps + 1} timesteps for future_steps={future_steps}."
            )

    logging.info("Data validation completed successfully.")


class GenericPhysicsDataset(Dataset):
    """
    A generic PyTorch Dataset for loading preprocessed physics data with sliding window sample generation
    and channel-wise normalization using precomputed min and max values.

    This class is designed to be flexible and can handle various datasets by specifying the data directories,
    number of channels, and other relevant parameters.
    """

    def __init__(
        self,
        data_dirs,
        future_steps=1,
        min_max_path=None,
        required_channels=None,
        validate=True,
    ):
        """
        Initializes the GenericPhysicsDataset.

        Args:
            data_dirs (list of str): List of directories containing preprocessed `.npy` files.
                                      Typically includes either train or test directories.
            future_steps (int): Number of timesteps in the future the model will predict.
            min_max_path (str, optional): Path to the JSON file containing min and max values for each channel.
                                          If None, it will look for 'min_max.json' in each data directory.
            required_channels (int, optional): Number of channels expected in the data.
                                               If None, it will be inferred from the min_max.json file.
            validate (bool, optional): Whether to perform data validation upon initialization. Defaults to True.
        """
        if validate:
            validate_data_format(
                data_dirs, future_steps, min_max_path, required_channels
            )

        self.data_dirs = data_dirs
        self.future_steps = future_steps
        self.files = []

        # Aggregate all .npy files from the specified directories
        for data_dir in data_dirs:
            dir_files = sorted(
                [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith(".npy")
                ]
            )
            self.files.extend(dir_files)

        # Load min and max values
        if min_max_path is None:
            # Assume min_max.json is present in each data directory
            min_max_files = [
                os.path.join(data_dir, "min_max.json")
                for data_dir in data_dirs
            ]
            # Merge min and max from all min_max.json files
            self.channel_min = []
            self.channel_max = []
            for mm_path in min_max_files:
                with open(mm_path, "r") as f:
                    min_max = json.load(f)
                self.channel_min.extend(min_max["channel_min"])
                self.channel_max.extend(min_max["channel_max"])
        else:
            with open(min_max_path, "r") as f:
                min_max = json.load(f)
            self.channel_min = min_max["channel_min"]
            self.channel_max = min_max["channel_max"]

        # Determine the number of channels
        self.num_channels = len(self.channel_min)

        # Precompute the number of samples across all files
        self.samples = []
        logging.info("Preparing dataset samples...")
        for file_idx, file in enumerate(
            tqdm(self.files, desc="Listing samples")
        ):
            data_memmap = np.load(file, mmap_mode="r")
            timesteps = data_memmap.shape[
                0
            ]  # Shape: (timesteps, channels, height, width)
            del data_memmap  # Close the memmap

            max_start_t = timesteps - self.future_steps - 1
            for start_t in range(0, max_start_t + 1):
                self.samples.append((file_idx, start_t))

        logging.info(f"Total samples in dataset: {len(self.samples)}")

        # Precompute t1 assuming all files have the same number of timesteps
        if len(self.files) > 0:
            sample_memmap = np.load(self.files[0], mmap_mode="r")
            timesteps = sample_memmap.shape[0]
            del sample_memmap
            whole_t = timesteps + 1  # As per original code
            self.t1 = torch.tensor(
                [(i + 1) / whole_t for i in range(self.future_steps)],
                dtype=torch.float32,
            )  # Shape: (future_steps,)
            self.t0 = torch.tensor(0.0, dtype=torch.float32)  # Scalar
        else:
            raise ValueError(
                "No valid .npy files found in the specified directories."
            )

        # Initialize a cache for memory-mapped files to improve performance
        self._memmap_cache = {}

    def __len__(self):
        return len(self.samples)

    def normalize_channel(self, tensor, channel_idx):
        """
        Normalizes a specific channel of the tensor between 0 and 1.

        Args:
            tensor (torch.Tensor): The tensor to normalize.
            channel_idx (int): The index of the channel to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        min_val = self.channel_min[channel_idx]
        max_val = self.channel_max[channel_idx]
        if max_val - min_val == 0:
            raise ValueError(
                f"Max and min values for channel {channel_idx} are the same. Cannot normalize."
            )
        tensor[:, channel_idx, :, :] = (
            tensor[:, channel_idx, :, :] - min_val
        ) / (max_val - min_val)
        return tensor

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (ic, t0, t1, target)
                - ic: Initial condition tensor of shape (channels, height, width)
                - t0: Scalar tensor (0.0)
                - t1: Tensor of shape (future_steps,)
                - target: Tensor of shape (future_steps, channels, height, width)
        """
        # Retrieve the (file_idx, start_t) tuple for this sample
        file_idx, start_t = self.samples[idx]
        file = self.files[file_idx]

        # Check if the memmap for this file is already cached
        if file not in self._memmap_cache:
            try:
                # Memory-map the file and store in cache
                data_memmap = np.load(file, mmap_mode="r")
                self._memmap_cache[file] = data_memmap
            except Exception as e:
                raise ValueError(f"Error loading file '{file}': {e}")

        data_memmap = self._memmap_cache[file]

        # Convert to PyTorch tensor
        try:
            # Access the required timesteps: start_t to start_t + 1 + future_steps
            required_timesteps = slice(
                start_t, start_t + 1 + self.future_steps
            )
            data = data_memmap[
                required_timesteps, :, :, :
            ]  # Shape: (future_steps +1, channels, height, width)
        except Exception as e:
            raise ValueError(
                f"Error accessing timesteps {start_t} to {start_t + self.future_steps +1} in file '{file}': {e}"
            )

        # Convert to PyTorch tensor
        data_tensor = torch.from_numpy(
            data.copy()
        ).float()  # Shape: (future_steps +1, channels, height, width)

        # Normalize each channel between 0 and 1 using precomputed min and max
        for channel_idx in range(self.num_channels):
            data_tensor = self.normalize_channel(data_tensor, channel_idx)

        # Extract input timestep
        ic = data_tensor[0]  # Shape: (channels, height, width)

        # Prepare the target sequence (ground truth)
        target = data_tensor[
            1:
        ]  # Shape: (future_steps, channels, height, width)

        return ic, self.t0, self.t1, target

    def __del__(self):
        # Close all memmap files when the dataset is deleted
        for memmap in self._memmap_cache.values():
            del memmap
        self._memmap_cache.clear()


def custom_collate_fn(batch):
    """
    Custom collate function to rearrange the target tensor.

    Args:
        batch: A list of tuples (ic, t0, t1, target)

    Returns:
        Batched tensors and fixed time indicators:
            - ic: (batch_size, channels, height, width)
            - t0: 0.0 (scalar tensor)
            - t1: (future_steps,) tensor
            - target: (future_steps, batch_size, channels, height, width)
    """
    ic, t0, t1, target = zip(*batch)

    # Stack the initial conditions into a tensor
    ic = torch.stack(ic, dim=0)  # Shape: (batch_size, channels, height, width)

    # Since t0 is always 0.0, return a single scalar tensor
    t0 = torch.tensor(0.0, dtype=torch.float32)  # Scalar tensor

    # Since t1 is consistent across all samples, take the first one
    t1 = t1[0]  # Shape: (future_steps,)

    # Stack targets into a tensor and permute to match desired shape
    target = torch.stack(target, dim=0).permute(
        1, 0, 2, 3, 4
    )  # Shape: (future_steps, batch_size, channels, height, width)

    return ic, t0, t1, target


class InitialConditionDataset(Dataset):
    """
    A PyTorch Dataset for loading only the initial condition (first time step) from preprocessed physics data.

    Each sample consists of:
        - ic: Initial condition tensor of shape (channels, height, width)
        - t0: Scalar tensor (0.0)
        - t1: Tensor indicating future steps (defined by the user or calculated automatically)
        - target: Placeholder or can be set to None since the model will predict the entire sequence
    """

    def __init__(
        self,
        data_dirs,
        future_steps=1,
        t1=None,
        min_max_path=None,
        required_channels=None,
        validate=True,
    ):
        """
        Initializes the InitialConditionDataset.

        Args:
            data_dirs (list of str): List of directories containing preprocessed `.npy` files.
            future_steps (int): Number of timesteps in the future the model will predict.
            t1 (list, optional): A list of future steps for prediction. If None, evenly spaced steps will be calculated.
            min_max_path (str, optional): Path to the JSON file containing min and max values for each channel.
                                          If None, it will look for 'min_max.json' in each data directory.
            required_channels (int, optional): Number of channels expected in the data.
                                               If None, it will be inferred from the min_max.json file.
            validate (bool, optional): Whether to perform data validation upon initialization. Defaults to True.
        """
        if validate:
            validate_data_format(
                data_dirs, future_steps, min_max_path, required_channels
            )

        self.data_dirs = data_dirs
        self.future_steps = future_steps
        self.files = []

        # Aggregate all .npy files from the specified directories
        for data_dir in data_dirs:
            dir_files = sorted(
                [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith(".npy")
                ]
            )
            if not dir_files:
                logging.warning(
                    f"No .npy files found in directory '{data_dir}'."
                )
            self.files.extend(dir_files)

        if not self.files:
            raise ValueError(
                "No .npy files found in any of the specified directories."
            )

        # Load min and max values
        if min_max_path is None:
            # Assume min_max.json is present in each data directory
            min_max_files = [
                os.path.join(data_dir, "min_max.json")
                for data_dir in data_dirs
            ]
            # Merge min and max from all min_max.json files
            self.channel_min = []
            self.channel_max = []
            for mm_path in min_max_files:
                if not os.path.exists(mm_path):
                    raise FileNotFoundError(
                        f"Min and max values file not found at '{mm_path}'. "
                        "Please ensure the file exists."
                    )
                with open(mm_path, "r") as f:
                    min_max = json.load(f)
                if (
                    "channel_min" not in min_max
                    or "channel_max" not in min_max
                ):
                    raise ValueError(
                        f"'channel_min' or 'channel_max' not found in '{mm_path}'."
                    )
                self.channel_min.extend(min_max["channel_min"])
                self.channel_max.extend(min_max["channel_max"])
        else:
            if not os.path.exists(min_max_path):
                raise FileNotFoundError(
                    f"Min and max values file not found at '{min_max_path}'. "
                    "Please ensure the file exists."
                )
            with open(min_max_path, "r") as f:
                min_max = json.load(f)
            if "channel_min" not in min_max or "channel_max" not in min_max:
                raise ValueError(
                    f"'channel_min' or 'channel_max' not found in '{min_max_path}'."
                )
            self.channel_min = min_max["channel_min"]
            self.channel_max = min_max["channel_max"]

        num_channels = len(self.channel_min)
        if len(self.channel_max) != num_channels:
            raise ValueError(
                "Length of 'channel_min' and 'channel_max' must be the same."
            )

        if required_channels is not None:
            if num_channels != required_channels:
                raise ValueError(
                    f"Number of channels in min_max.json ({num_channels}) does not match "
                    f"the required_channels ({required_channels})."
                )
            logging.info(f"Number of channels validated: {num_channels}")

        self.num_channels = num_channels

        # Validate that each file has at least one timestep
        logging.info("Validating timesteps in each file...")
        self.timesteps_per_file = []
        for file in tqdm(self.files, desc="Validating timesteps"):
            try:
                data = np.load(file, mmap_mode="r")
            except Exception as e:
                raise ValueError(f"Error loading file '{file}': {e}")

            timesteps, channels, height, width = data.shape
            if channels != self.num_channels:
                raise ValueError(
                    f"File '{file}' has {channels} channels; expected {self.num_channels} channels based on min_max.json."
                )
            if timesteps < 1:
                raise ValueError(
                    f"File '{file}' has {timesteps} timesteps; requires at least 1 timestep."
                )
            self.timesteps_per_file.append(timesteps)

        # Handle t1 logic: either use the user-defined list or generate evenly spaced steps
        if t1 is None:
            self.t1 = torch.tensor(
                [
                    (i + 1) / self.future_steps
                    for i in range(self.future_steps)
                ],
                dtype=torch.float32,
            )  # Shape: (future_steps,)
        else:
            if len(t1) != future_steps:
                raise ValueError(
                    f"Length of 't1' ({len(t1)}) must match 'future_steps' ({future_steps})."
                )
            self.t1 = torch.tensor(t1, dtype=torch.float32)  # User-defined t1

        self.t0 = torch.tensor(0.0, dtype=torch.float32)  # Scalar

        # Initialize a cache for memory-mapped files to improve performance
        self._memmap_cache = {}

    def __len__(self):
        return len(self.files)

    def normalize_channel(self, tensor, channel_idx):
        """
        Normalizes a specific channel of the tensor between 0 and 1.

        Args:
            tensor (torch.Tensor): The tensor to normalize.
            channel_idx (int): The index of the channel to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        min_val = self.channel_min[channel_idx]
        max_val = self.channel_max[channel_idx]
        if max_val - min_val == 0:
            raise ValueError(
                f"Max and min values for channel {channel_idx} are the same. Cannot normalize."
            )
        tensor[channel_idx, :, :] = (tensor[channel_idx, :, :] - min_val) / (
            max_val - min_val
        )
        return tensor

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (ic, t0, t1)
                - ic: Initial condition tensor of shape (channels, height, width)
                - t0: Scalar tensor (0.0)
                - t1: Tensor of shape (future_steps,)
        """
        file = self.files[idx]

        # Check if the memmap for this file is already cached
        if file not in self._memmap_cache:
            try:
                # Memory-map the file and store in cache
                data_memmap = np.load(file, mmap_mode="r")
                self._memmap_cache[file] = data_memmap
            except Exception as e:
                raise ValueError(f"Error loading file '{file}': {e}")

        data_memmap = self._memmap_cache[file]

        # Convert to PyTorch tensor
        try:
            # Access the first timestep
            data = data_memmap[0, :, :, :]  # Shape: (channels, height, width)
        except Exception as e:
            raise ValueError(
                f"Error accessing the first timestep in file '{file}': {e}"
            )

        # Convert to PyTorch tensor
        data_tensor = torch.from_numpy(
            data.copy()
        ).float()  # Shape: (channels, height, width)

        # Normalize each channel between 0 and 1 using precomputed min and max
        for channel_idx in range(self.num_channels):
            data_tensor = self.normalize_channel(data_tensor, channel_idx)

        ic = data_tensor  # Shape: (channels, height, width)

        return (
            ic,
            self.t0,
            self.t1,
            None,
        )  # Target is None since the model will predict it


def initial_condition_collate_fn(batch):
    """
    Custom collate function for InitialConditionDataset.

    Args:
        batch: A list of tuples (ic, t0, t1, target)

    Returns:
        tuple: Batched tensors and fixed time indicators:
            - ic: (batch_size, channels, height, width)
            - t0: Scalar tensor (0.0)
            - t1: (future_steps,) tensor
            - target: None (since target is not used)
    """
    ic, t0, t1, target = zip(*batch)

    # Stack the initial conditions into a tensor
    ic = torch.stack(ic, dim=0)  # Shape: (batch_size, channels, height, width)

    # Since t0 is always 0.0, return a single scalar tensor
    t0 = torch.tensor(0.0, dtype=torch.float32)  # Scalar tensor

    # Since t1 is consistent across all samples, take the first one
    t1 = t1[0]  # Shape: (future_steps,)

    # Targets are None, so we can return None or handle accordingly
    target = None

    return ic, t0, t1, target
