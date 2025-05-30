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
        shape = (2, 4, 4, 4)  # (timesteps, channels, H, W)
        values_list = [
            [1.0, 2.0, 3.0, 4.0],  # ch2 = 3.0, ch3 = 4.0 -> norm ~5.0
            [0.0, 5.0, 8.0, 6.0],  # ch2 = 8.0, ch3 = 6.0 -> norm ~10.0
            [3.0, 1.0, 7.0, 9.0],  # ch2 = 7.0, ch3 = 9.0 -> norm ~11.4
        ]

        # Save test files
        for i, values in enumerate(values_list):
            path = os.path.join(tmpdir, f"data_{i}.npy")
            create_test_npy(path, shape, values)

        output_path = os.path.join(tmpdir, "min_max.json")
        result = compute_min_max([tmpdir], output_file=output_path)

        # Load and verify
        with open(output_path, "r") as f:
            data = json.load(f)

        channel_min = data["channel_min"]
        channel_max = data["channel_max"]

        # Channels 0 and 1: basic check
        assert channel_min[0] == 0.0
        assert channel_max[0] == 3.0
        assert channel_min[1] == 1.0
        assert channel_max[1] == 5.0

        # Expected velocity norms (same everywhere in each file)
        expected_norms = [
            np.sqrt(3**2 + 4**2),   # 5.0
            np.sqrt(8**2 + 6**2),   # 10.0
            np.sqrt(7**2 + 9**2),   # ~11.401
        ]
        expected_min_norm = min(expected_norms)
        expected_max_norm = max(expected_norms)

        assert pytest.approx(channel_min[2], abs=1e-3) == expected_min_norm
        assert pytest.approx(channel_min[3], abs=1e-3) == expected_min_norm
        assert pytest.approx(channel_max[2], abs=1e-3) == expected_max_norm
        assert pytest.approx(channel_max[3], abs=1e-3) == expected_max_norm

        # Check vector with largest norm
        max_vec = result["max_norm_velocity_vector"]
        assert isinstance(max_vec, list)
        assert len(max_vec) == 2  # channel 2 and channel 3
        assert pytest.approx(np.linalg.norm(max_vec), abs=1e-3) == expected_max_norm
