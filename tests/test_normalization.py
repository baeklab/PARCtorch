import os
import json
import numpy as np
import tempfile
from unittest import mock
import pytest
from PARCtorch.data.normalization import compute_min_max

def create_test_npy(path, shape, fill_values_per_channel):
    """Helper to write a 4D .npy file with different values per channel."""
    array = np.zeros(shape, dtype=np.float32)
    for ch, val in enumerate(fill_values_per_channel):
        array[:, ch, :, :] = val
    np.save(path, array)

@mock.patch("builtins.print")  # Mute prints during test
def test_compute_min_max_combined_last_two_channel_max(mock_print):
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (2, 4, 4, 4)  # small shape for test: (timesteps, channels, height, width)

        # Each entry is [ch0, ch1, ch2, ch3] per file
        values_list = [
            [1.0, 2.0, 3.0, 4.0],  # Max in last two = 4.0
            [0.0, 5.0, 8.0, 6.0],  # Max in last two = 8.0
            [3.0, 1.0, 7.0, 9.0],  # Max in last two = 9.0 -> global max
        ]

        for i, values in enumerate(values_list):
            path = os.path.join(tmpdir, f"data_{i}.npy")
            create_test_npy(path, shape, values)

        output_path = os.path.join(tmpdir, "min_max.json")
        compute_min_max([tmpdir], output_file=output_path)

        # Load and verify
        with open(output_path, "r") as f:
            data = json.load(f)

        channel_min = data["channel_min"]
        channel_max = data["channel_max"]

        # Channels 0 and 1: standard behavior
        assert channel_min[0] == 0.0  # min of [1.0, 0.0, 3.0]
        assert channel_max[0] == 3.0
        assert channel_min[1] == 1.0  # min of [2.0, 5.0, 1.0]
        assert channel_max[1] == 5.0

        # Channels 2 and 3: special logic
        assert channel_min[2] == 5 # min of sqrt(9 + 16), sqrt(64 + 36), sqrt(49 + 81)
        assert channel_min[3] == 5 # min of sqrt(9 + 16), sqrt(64 + 36), sqrt(49 + 81)

        assert pytest.approx(channel_max[2], 1e-3) == 11.4 # max of sqrt(9 + 16), sqrt(64 + 36), sqrt(49 + 81)
        assert pytest.approx(channel_max[3], 1e-3) == 11.4# max of sqrt(9 + 16), sqrt(64 + 36), sqrt(49 + 81)
