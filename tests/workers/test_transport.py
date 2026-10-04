"""CPU tensor-serialization checks for the model-process boundary."""
import numpy as np
import torch
from python.fireprot_workers.transport import save_arrays


def test_bfloat16_is_saved_as_raw_bits_with_original_dtype(tmp_path):
    value = torch.tensor([0.0, -0.0, 1.0, -2.0, 1.0078125], dtype=torch.bfloat16)
    path = tmp_path / 'raw.npz'
    descriptors = save_arrays(path, {'coordinates': value})
    assert descriptors['coordinates'] == dict(shape=[5], original_dtype='bfloat16', storage_dtype='uint16')
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive['coordinates'], value.view(torch.uint16).numpy())
        assert archive['coordinates'][0] != archive['coordinates'][1]


def test_noncontiguous_values_keep_logical_shape_and_values(tmp_path):
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    path = tmp_path / 'raw.npz'
    descriptors = save_arrays(path, {'value': value, 'scalar': 0.5})
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive['value'], value.numpy())
        assert archive['scalar'].shape == ()
    assert descriptors['value']['shape'] == [4, 3]
    assert descriptors['value']['original_dtype'] == 'float32'
