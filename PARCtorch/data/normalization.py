# /Parctorch/Data/Normalization.py

import os
import json
import numpy as np


def compute_min_max(data_dirs, output_file="min_max.json"):
    """
    Computes the min and max values for each channel across multiple datasets.

    Args:
        data_dirs (list of str): List of directories containing `.npy` files.
        output_file (str): Name of the output JSON file.

    Returns:
        None
    """
    # Aggregate all .npy files from the specified directories
    all_files = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            raise ValueError(
                f"Data directory '{data_dir}' does not exist or is not a directory."
            )
        files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npy")
            ]
        )
        if not files:
            print(f"Warning: No .npy files found in directory '{data_dir}'.")
        all_files.extend(files)

    if not all_files:
        raise ValueError(
            "No .npy files found in any of the specified directories."
        )

    # Initialize min and max lists dynamically based on the first file
    try:
        sample_data = np.load(all_files[0])
    except Exception as e:
        raise ValueError(f"Error loading file '{all_files[0]}': {e}") from e

    if sample_data.ndim != 4:
        raise ValueError(
            f"Data in '{all_files[0]}' is expected to have 4 dimensions (timesteps, channels, height, width)."
        )

    _, channels, _, _ = sample_data.shape
    channel_min = [np.inf] * channels
    channel_max = [-np.inf] * channels

    print("Calculating channel-wise min and max values for normalization...")
    print("Current working directory:", os.getcwd())
    print(f"Total files to process: {len(all_files)}")

    for file_idx, file in enumerate(all_files):
        try:
            data = np.load(file)  # Shape: (timesteps, channels, height, width)
            if data.ndim != 4:
                raise ValueError(
                    f"Data in '{file}' does not have 4 dimensions."
                )
            _, file_channels, _, _ = data.shape
            if file_channels != channels:
                raise ValueError(
                    f"File '{file}' has {file_channels} channels, expected {channels}."
                )
        except Exception as e:
            print(f"Error loading file '{file}': {e}. Skipping this file.")
            continue

        # Iterate over each channel to find min and max
        for channel_idx in range(channels):
            channel_data = data[:, channel_idx, :, :]
            current_min = channel_data.min()
            current_max = channel_data.max()
            if current_min < channel_min[channel_idx]:
                channel_min[channel_idx] = current_min
            if current_max > channel_max[channel_idx]:
                channel_max[channel_idx] = current_max

        # Provide progress updates every 100 files or at the end
        if (file_idx + 1) % 100 == 0 or (file_idx + 1) == len(all_files):
            print(f"Processed {file_idx + 1}/{len(all_files)} files.")

    print("Channel-wise min values:", channel_min)
    print("Channel-wise max values:", channel_max)

    # Convert NumPy floats to Python floats for JSON serialization
    min_max = {
        "channel_min": [float(x) for x in channel_min],
        "channel_max": [float(x) for x in channel_max],
    }

    # Save the min and max values to a JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(min_max, f, indent=4)
        print(f"Min and max values saved to '{os.path.abspath(output_file)}'.")
    except Exception as e:
        raise IOError(
            f"Failed to write min and max values to '{output_file}': {e}"
        ) from e
