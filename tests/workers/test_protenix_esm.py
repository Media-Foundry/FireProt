"""Mini-ESM checkpoint and actual feature checks without upstream model loading."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from python.fireprot_workers.protenix import check_request
from python.fireprot_workers.protenix_esm import validate_features
from test_protenix import request


def esm_request(tmp_path):
    r = request(tmp_path)
    r['workload']['model_name'] = 'protenix_mini_esm_v0.5.0'
    r['checkpoints'] = {role: {'path': str(tmp_path / name)} for role, name in (
        ('model', 'protenix_mini_esm_v0.5.0.pt'),
        ('language_model', 'esm2_t36_3B_UR50D.pt'),
        ('language_model_regression', 'esm2_t36_3B_UR50D-contact-regression.pt'))}
    return r


def test_complete_esm_request_is_supported(tmp_path):
    assert check_request(esm_request(tmp_path))[1].name == 'protenix_mini_esm_v0.5.0.pt'


@pytest.mark.parametrize('role', ['language_model', 'language_model_regression'])
@pytest.mark.parametrize('mode', ['missing', 'different_directory', 'wrong_filename'])
def test_esm_never_reads_undeclared_sibling_weights(tmp_path, role, mode):
    r = esm_request(tmp_path)
    if mode == 'missing':
        del r['checkpoints'][role]
    else:
        path = Path(r['checkpoints'][role]['path'])
        r['checkpoints'][role]['path'] = str(tmp_path / 'other' / path.name if mode == 'different_directory' else tmp_path / 'other.pt')
    with pytest.raises(ValueError, match='co-located'):
        check_request(r)


def test_esm_truncation_is_rejected(tmp_path):
    r = esm_request(tmp_path)
    p = Path(r['input']['path'])
    data = json.loads(p.read_text())
    data[0]['sequences'][0]['proteinChain']['sequence'] = 'A' * 4095
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='truncated'):
        check_request(r)


@pytest.mark.parametrize('mode', ['valid', 'missing', 'zeros_fallback', 'truncated', 'dtype', 'nan', 'error_file'])
def test_actual_model_input_must_match_complete_generated_embedding(tmp_path, mode):
    r = esm_request(tmp_path)
    root = tmp_path / 'esm_embeddings'
    p = root / 'esm2-3b/protein_1/protein_1.pt'
    p.parent.mkdir(parents=True)
    value = torch.ones(6, 2560)
    feature = value.clone()
    if mode == 'truncated': value = value[:-1]
    elif mode == 'dtype': value = value.to(torch.bfloat16)
    elif mode == 'nan': value[0, 0] = float('nan')
    elif mode == 'zeros_fallback': feature.zero_()
    elif mode == 'error_file':
        (root / 'esm_error').mkdir()
        (root / 'esm_error/protein_1.txt').write_text('upstream failure')
    if mode != 'missing': torch.save(value, p)
    data = {'input_feature_dict': {'esm_token_embedding': feature}}
    config = SimpleNamespace(enable=True, model_name='esm2-3b', embedding_dim=2560)
    if mode == 'valid':
        assert validate_features(r, data, config, tmp_path) is feature
    else:
        with pytest.raises((RuntimeError, FileNotFoundError)):
            validate_features(r, data, config, tmp_path)
