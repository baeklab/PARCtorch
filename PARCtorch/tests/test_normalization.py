import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import json
import numpy as np
import tempfile
import pytest
from unittest import mock
from PARCtorch.data.normalization import compute_min_max

def create_test_npy(path, shape, fill_values_per_channel):
    """Helper to write a 4D .npy file with different values per channel."""
    array = np.zeros(shape, dtype=np.float32)
    for ch, val in enumerate(fill_values_per_channel):
        array[:, ch, :, :] = val
    np.save(path, array)

@mock.patch("builtins.print")  # Mute prints during test
def test_compute_min_max_special_channel_logic(mock_print):
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (2, 4, 8, 8)  # (timesteps, channels, height, width)

        # Create files with distinct values for each channel
        values_list = [
            [1.0, 2.0, 3.0, 4.0],  # File 1
            [0.0, 5.0, 8.0, 6.0],  # File 2
            [3.0, 1.0, 7.0, 9.0],  # File 3
        ]

        for i, values in enumerate(values_list):
            path = os.path.join(tmpdir, f"data_{i}.npy")
            create_test_npy(path, shape, values)

        output_path = os.path.join(tmpdir, "min_max.json")
        compute_min_max([tmpdir], output_file=output_path)

        # Check output file
        assert os.path.isfile(output_path)

        with open(output_path, "r") as f:
            data = json.load(f)

        channel_min = data["channel_min"]
        channel_max = data["channel_max"]

        # Standard channels (0 and 1)
        expected_min = [0.0, 1.0]  # channel 0 min across [1.0, 0.0, 3.0], channel 1 min across [2.0, 5.0, 1.0]
        expected_max = [3.0, 5.0]

        assert channel_min[0] == expected_min[0]
        assert channel_min[1] == expected_min[1]
        assert channel_max[0] == expected_max[0]
        assert channel_max[1] == expected_max[1]

        # Special channels (2 and 3) — min should be 0, max should be actual max per channel
        assert channel_min[2] == 0.0
        assert channel_min[3] == 0.0
        assert channel_max[2] == max([3.0, 8.0, 7.0])
        assert channel_max[3] == max([4.0, 6.0, 9.0])
