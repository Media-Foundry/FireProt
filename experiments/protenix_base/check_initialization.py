"""CPU-only Base construction qualification; does not enable production dispatch."""
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from ml_collections import ConfigDict
from scipy.stats import truncnorm

from configs.configs_base import configs
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from protenix.model.protenix import Protenix
from protenix.model.modules import primitives
from protenix.model.triangular import layers
from fireprot.adapters.protenix import construction


def main(checkpoint, destination):
    torch.set_num_threads(1)
    cfg = parse_configs(configs=copy.deepcopy({**configs, 'data': data_configs,
                        **inference_configs}), fill_required_with_null=True)
    cfg.update(ConfigDict(model_configs['protenix_base_default_v0.5.0']))
    cfg.model_name = 'protenix_base_default_v0.5.0'
    cfg.dtype = 'fp32'
    cfg.triangle_attention = cfg.triangle_multiplicative = 'torch'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    before = construction._capture_rng('cpu')
    start = time.perf_counter()
    reference = Protenix(cfg)
    native_seconds = time.perf_counter() - start
    reference.load_state_dict(state, strict=True)
    after = construction._capture_rng('cpu')
    construction._restore_rng(before)
    statistics = dict(uniform_calls=0, uniform_elements=0, native_initializer_calls=0)
    start = time.perf_counter()
    with construction._defer_initializers(layers, primitives, truncnorm, statistics):
        candidate = Protenix(cfg)
    candidate_seconds = time.perf_counter() - start
    candidate.load_state_dict(state, strict=True)
    actual = construction._capture_rng('cpu')
    rng_equal = (after['python'] == actual['python']
                 and after['numpy'][0] == actual['numpy'][0]
                 and np.array_equal(after['numpy'][1], actual['numpy'][1])
                 and after['numpy'][2:] == actual['numpy'][2:]
                 and torch.equal(after['cpu'], actual['cpu']))
    parameters_equal = all(torch.equal(value, candidate.state_dict()[key])
                           for key, value in reference.state_dict().items())
    buffers_equal = all(torch.equal(value, dict(candidate.named_buffers())[key])
                        for key, value in reference.named_buffers())
    report = dict(model=cfg.model_name, device='cpu', strict_load=True,
                  rng_equal=rng_equal, parameters_equal=parameters_equal,
                  buffers_equal=buffers_equal, native_construction_seconds=native_seconds,
                  candidate_construction_seconds=candidate_seconds, statistics=statistics,
                  passed=rng_equal and parameters_equal and buffers_equal)
    Path(destination).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main(*sys.argv[1:])
