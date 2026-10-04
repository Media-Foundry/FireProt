"""Fail before model loading for inputs/assets outside the integrated contract."""
import copy
import json

import pytest

from python.fireprot_workers.protenix import RESOURCE_NAMES, check_request


def request(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'name': 'protein', 'sequences': [
        {'proteinChain': {'sequence': 'ACDEFX', 'count': 1}}]}]))
    return dict(source={'commit': 'bb3be993b52713795796c72cf0a46ef8a7323442'},
                workload={'model_name': 'protenix_mini_default_v0.5.0', 'precision': 'float32'},
                resources={key: {'path': str(tmp_path / key)} for key in RESOURCE_NAMES},
                checkpoints={'model': {'path': str(tmp_path / 'protenix_mini_default_v0.5.0.pt')}},
                input={'path': str(path)}, seed=101)


def test_pinned_sequence_only_request(tmp_path):
    r = request(tmp_path)
    name, checkpoint = check_request(r)
    assert name == 'protein' and checkpoint.name == 'protenix_mini_default_v0.5.0.pt'


@pytest.mark.parametrize('change', ['source', 'model', 'precision', 'resource', 'checkpoint', 'seed'])
def test_incompatible_request_rejected_before_model_import(tmp_path, change):
    r = request(tmp_path)
    if change == 'source': r['source']['commit'] = 'a' * 40
    elif change == 'model': r['workload']['model_name'] = 'protenix_mini_esm_v0.5.0'
    elif change == 'precision': r['workload']['precision'] = 'bfloat16'
    elif change == 'resource': r['resources'].pop(RESOURCE_NAMES[0])
    elif change == 'checkpoint': r['checkpoints']['model']['path'] = 'other.pt'
    else: r['seed'] = 2**32
    with pytest.raises(ValueError): check_request(r)


@pytest.mark.parametrize('change', ['escape', 'seeds', 'feature', 'copies', 'multiple'])
def test_external_or_ambiguous_input_not_silently_reduced(tmp_path, change):
    r = request(tmp_path)
    path = tmp_path / 'input.json'
    data = json.loads(path.read_text())
    if change == 'escape': data[0]['name'] = '../../outside'
    elif change == 'seeds': data[0]['modelSeeds'] = [102]
    elif change == 'feature': data[0]['sequences'][0]['proteinChain']['msa'] = {'path': '/external'}
    elif change == 'copies': data[0]['sequences'][0]['proteinChain']['count'] = 2
    else: data.append(copy.deepcopy(data[0]))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): check_request(r)


def test_distributed_environment_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError, match='Distributed'): check_request(request(tmp_path))


def test_tiny_default_uses_its_own_checkpoint_and_no_language_model(tmp_path):
    r = request(tmp_path)
    r['workload']['model_name'] = 'protenix_tiny_default_v0.5.0'
    with pytest.raises(ValueError, match='Checkpoint filename'):
        check_request(r)
    r['checkpoints']['model']['path'] = str(tmp_path / 'protenix_tiny_default_v0.5.0.pt')
    assert check_request(r)[1].name == 'protenix_tiny_default_v0.5.0.pt'
    r['checkpoints']['language_model'] = {'path': 'unused.pt'}
    with pytest.raises(ValueError, match='Unexpected Protenix checkpoint roles'):
        check_request(r)
