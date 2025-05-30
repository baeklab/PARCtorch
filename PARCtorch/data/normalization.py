# /Parctorch/Data/Normalization.py

import os
import json
import numpy as np


def compute_min_max(data_dirs, output_file="min_max.json"):
    """
    Computes the min, max, mean, and standard deviation values for each channel across multiple datasets.
    Also computes the mean velocity field and the velocity vector with the highest magnitude
    using the last two channels (assumed to represent velocity components).

    Args:
        data_dirs (list of str): List of directories containing `.npy` files.
        output_file (str): Name of the output JSON file.

    Returns:
        dict: Statistics including min/max/mean/std per channel, mean velocity vector,
              and the velocity vector with the highest norm.
    """
    
    # Aggregate all .npy files from provided directories
    all_files = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            raise ValueError(f"Data directory '{data_dir}' does not exist or is not a directory.")
        
        # Collect sorted list of .npy files
        files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ])
        
        if not files:
            print(f"Warning: No .npy files found in directory '{data_dir}'.")
        all_files.extend(files)

    if not all_files:
        raise ValueError("No .npy files found in any of the specified directories.")

    # Load first file to determine shape and number of channels
    try:
        sample_data = np.load(all_files[0])
    except Exception as e:
        raise ValueError(f"Error loading file '{all_files[0]}': {e}") from e

    if sample_data.ndim != 4:
        raise ValueError("Expected 4D data (timesteps, channels, height, width).")

    _, channels, height, width = sample_data.shape

    # Initialize tracking variables for global statistics
    channel_min = [np.inf] * channels
    channel_max = [-np.inf] * channels
    channel_sum = [0.0] * channels
    channel_sum_sq = [0.0] * channels
    total_count = [0] * channels

    min_norm = np.inf
    max_norm = -np.inf
    max_norm_vector = None
    velocity_sum = np.zeros((2, height, width))

    print("Calculating channel-wise min and max values for normalization...")
    print(f"Total files to process: {len(all_files)}")

    for file_idx, file in enumerate(all_files):
        # Load each file and validate dimensions
        try:
            data = np.load(file)
            if data.ndim != 4:
                raise ValueError(f"Data in '{file}' does not have 4 dimensions.")
            _, file_channels, _, _ = data.shape
            if file_channels != channels:
                raise ValueError(f"File '{file}' has {file_channels} channels, expected {channels}.")
        except Exception as e:
            print(f"Error loading file '{file}': {e}. Skipping this file.")
            continue

        # Accumulate per-channel statistics
        for channel_idx in range(channels):
            channel_data = data[:, channel_idx, :, :]

            # Update min and max for this channel
            channel_min[channel_idx] = min(channel_min[channel_idx], channel_data.min())
            channel_max[channel_idx] = max(channel_max[channel_idx], channel_data.max())

            # Accumulate for mean and std computation
            channel_sum[channel_idx] += channel_data.sum()
            channel_sum_sq[channel_idx] += np.square(channel_data).sum()
            total_count[channel_idx] += channel_data.size

        # Compute velocity norms using last two channels (assumed to be velocity)
        velocity_data = data[:, -2:, :, :]  # Shape: (T, 2, H, W)
        velocity_sum += velocity_data.sum(axis=0)  # Sum across timesteps

        # Norm per spatial position per timestep
        norms = np.linalg.norm(velocity_data, axis=1)  # Shape: (T, H, W)
        max_idx = np.unravel_index(np.argmax(norms), norms.shape)
        current_max_vector = velocity_data[max_idx[0], :, max_idx[1], max_idx[2]]
        current_max_norm = norms[max_idx]

        # Update global max and min norm
        if current_max_norm > max_norm:
            max_norm = current_max_norm
            max_norm_vector = current_max_vector.copy()

        min_norm = min(min_norm, norms.min())

        # Print progress every 100 files
        if (file_idx + 1) % 100 == 0 or (file_idx + 1) == len(all_files):
            print(f"Processed {file_idx + 1}/{len(all_files)} files.")

    # Override min/max for last two channels (velocity)
    channel_min[-2] = channel_min[-1] = min_norm
    channel_max[-2] = channel_max[-1] = max_norm

    # Finalize mean and std dev
    # Compute channel mean: s is sum , c is count, mean = sum / count
    channel_mean = [s / c if c > 0 else 0.0 for s, c in zip(channel_sum, total_count)]

    # Comput channel std dev: 
    # sq: Accumulated total of squares (i.e., sum(x^2))
    # c: count of elements
    # m : mean computed above

    channel_std_dev = [
        np.sqrt((sq / c) - (m ** 2)) if c > 0 else 0.0 # E[X^2] - (E[X])^2
        for sq, c, m in zip(channel_sum_sq, total_count, channel_mean)
    ]

    # Mean velocity vector over all samples
    velocity_total_count = sum(total_count[-2:]) / 2  # each velocity channel contributes
    mean_velocity = velocity_sum / velocity_total_count if velocity_total_count > 0 else None

    # Assemble result dictionary for saving and returning
    result = {
        "channel_min": [float(x) for x in channel_min],
        "channel_max": [float(x) for x in channel_max],
        "channel_mean": [float(x) for x in channel_mean],
        "channel_std_dev": [float(x) for x in channel_std_dev],
        "mean_velocity_vector": mean_velocity.tolist() if mean_velocity is not None else None,
        "max_norm_velocity_vector": max_norm_vector.tolist() if max_norm_vector is not None else None,
    }

    # Write result to JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Min and max values saved to '{os.path.abspath(output_file)}'.")
    except Exception as e:
        raise IOError(f"Failed to write to '{output_file}': {e}") from e

    return result
