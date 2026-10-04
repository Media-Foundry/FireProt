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


def test_input_snapshot_is_announced_with_its_content_hash(tmp_path, monkeypatch):
    import hashlib
    from python.fireprot_workers.transport import announce_outputs
    monkeypatch.setenv('FIREPROT_OUTPUT_DIR', str(tmp_path))
    snapshot = tmp_path / 'input-snapshot.fasta'
    snapshot.write_bytes(b'>fixture\nACDX\n')
    events = []
    channel = type('FixtureChannel', (), {'emit': lambda self, event: events.append(event)})()
    announce_outputs(channel, dict(schema_version=3, predictions=[], input_snapshot=snapshot.name))
    assert dict(path=snapshot.name, sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()) in events[0]['artifacts']
