# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for FireProt: extracted noisy-coordinate projection with rollout-local
# atom conditioning reuse. This entire file is Apache-2.0.

"""Experimental compute-skipping cache for the audited immutable OF3 rollout.

Source-pinned inputs, parameters and consumers must remain immutable. Identity
guards detect replacements, not in-place mutation of inference tensors. The
verified_atom_reuse diagnostic established these invariants for the target runs;
this cache still requires independent complete-model acceptance.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path

import torch

from .pair_cache import dependency_key
from .verified_atom_reuse import validate_atom_sources


def eligible(inputs):
    rl, single, pair = (inputs[k] for k in ('rl', 'si_trunk', 'zij_trunk'))
    if any(type(t) is not torch.Tensor for t in (rl, single, pair)):
        return False
    if any(t.dtype != torch.float32 or not t.is_cuda or t.device != rl.device
           or t.requires_grad or t.is_neg() or t.is_conj() for t in (rl, single, pair)):
        return False
    return (rl.ndim == 4 and tuple(rl.shape[:2]) == (1, 1) and rl.shape[-1] == 3
            and single.ndim == 4 and tuple(single.shape[:2]) == (1, 1)
            and pair.ndim == 5 and tuple(pair.shape[:2]) == (1, 1)
            and pair.shape[-1] == 128 and pair.shape[-2] > 0
            and pair.shape[-2] == pair.shape[-3] == single.shape[-2])


@contextmanager
def atom_cache(sample, output):
    encoder = sample.diffusion_module.atom_attn_enc
    validate_atom_sources(sample, encoder)
    signature = inspect.signature(encoder.get_atom_reps)
    originals = []
    state = None
    report = dict(schema_version=1, mode='experimental_atom_cache', scopes=[],
                  limits=['Audited immutable rollout only; no in-place mutation detection.',
                          'Retained bytes describe tensors, not whole-model peak memory.'])

    def replace(obj, name, factory):
        originals.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, factory(getattr(obj, name)))

    def wrap_rollout(original):
        def call(*args, **kwargs):
            nonlocal state
            if state is not None:
                raise RuntimeError('Nested atom cache rollout')
            if (encoder.training or torch.is_grad_enabled()
                    or not torch.are_deterministic_algorithms_enabled()
                    or torch.is_deterministic_algorithms_warn_only_enabled()):
                raise RuntimeError('Atom cache requires strict deterministic inference')
            record = dict(calls=0, hits=0, misses=0, fallbacks=0, completed=False)
            report['scopes'].append(record)
            state = dict(record=record, cached=None, key=None, owners=None)
            try:
                result = original(*args, **kwargs)
                record['completed'] = True
                return result
            finally:
                state = None
        return call

    def wrap_atoms(original):
        def call(*args, **kwargs):
            if state is None:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = bound.arguments
            record = state['record']
            record['calls'] += 1
            if not eligible(inputs):
                state.update(cached=None, key=None, owners=None)
                record['fallbacks'] += 1
                return original(*args, **kwargs)
            rl = inputs['rl']
            dependencies = {k: v for k, v in inputs.items() if k != 'rl'}
            key = (dependency_key(dependencies), tuple(rl.shape), tuple(rl.stride()),
                   rl.dtype, rl.device)
            if state['cached'] is not None and state['key'] == key:
                cl, plm = state['cached']
                ql = cl + encoder.noisy_position_embedder.linear_r(rl)
                record['hits'] += 1
                return ql, cl, plm
            state.update(cached=None, key=None, owners=None)
            ql, cl, plm = original(*args, **kwargs)
            state.update(cached=(cl, plm), key=key, owners=dependencies)
            record['misses'] += 1
            record['retained_bytes'] = sum(t.numel() * t.element_size() for t in (cl, plm))
            return ql, cl, plm
        return call

    try:
        replace(sample, '_sample_rollout', wrap_rollout)
        replace(encoder, 'get_atom_reps', wrap_atoms)
        yield report
    finally:
        for obj, name, had_local, local in reversed(originals):
            if had_local:
                setattr(obj, name, local)
            else:
                delattr(obj, name)
        Path(output, 'atom-cache.json').write_text(json.dumps(report, indent=2) + '\n')
