"""Pinned upstream Mini-ESM assets and tensor boundary; no shared cache policy."""
import json
from pathlib import Path


def check_checkpoints(request, checkpoint):
    expected = {'model'}
    if request['workload']['model_name'] == 'protenix_mini_esm_v0.5.0':
        for role, filename in (
                ('language_model', 'esm2_t36_3B_UR50D.pt'),
                ('language_model_regression', 'esm2_t36_3B_UR50D-contact-regression.pt')):
            expected.add(role)
            entry = request['checkpoints'].get(role)
            if entry is None or Path(entry['path']) != checkpoint.parent / filename:
                raise ValueError(f'Mini-ESM requires co-located pinned {role}: {filename}')
    if set(request['checkpoints']) != expected:
        raise ValueError('Unexpected Protenix checkpoint roles')


def validate_features(request, data, config, output):
    """Reject swallowed featurizer failures and retain the exact model input."""
    import torch
    if not config.enable or config.model_name != 'esm2-3b' or config.embedding_dim != 2560:
        raise RuntimeError('Unexpected Mini-ESM language model configuration')
    record, = json.loads(Path(request['input']['path']).read_text())
    sequence = record['sequences'][0]['proteinChain']['sequence']
    label = record['name'] + '_1'
    root = output / 'esm_embeddings'
    cached = torch.load(root / 'esm2-3b' / label / (label + '.pt'), weights_only=True, map_location='cpu')
    feature = data['input_feature_dict'].get('esm_token_embedding')
    if (not isinstance(feature, torch.Tensor) or not isinstance(cached, torch.Tensor)
            or feature.dtype != torch.float32 or cached.dtype != torch.float32
            or tuple(feature.shape) != (len(sequence), 2560) or cached.shape != feature.shape
            or not torch.isfinite(cached).all() or not torch.equal(feature.cpu(), cached)
            or any((root / 'esm_error').glob('*.txt'))):
        raise RuntimeError('Mini-ESM input does not match fresh complete upstream embeddings')
    return feature
