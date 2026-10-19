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
# Modified for FireProt: extracted noisy-coordinate projection and verified
# rollout-local reuse of atom conditioning. This entire file is Apache-2.0.

"""Mandatory native recomputation for proposed atom-conditioning reuse.

Every call verifies cached cl/plm and the recomputed noisy ql against upstream.
There is no unchecked or compute-skipping mode. Not performance evidence.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

import torch

from .verified_reuse import same_bits


def validate_atom_sources(sample, encoder):
    for obj, expected in (
        (sample, '3b5ed9fb090d53df2b5ad16be517cfdca7f5290a6bb8274e71408a3ff465155d'),
        (encoder, '3757d143e74153f1543fd4ec53891ff4a6c4d57d95ca5b7c22c4714ecc6890be'),
    ):
        if hashlib.sha256(Path(inspect.getfile(type(obj))).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Atom reuse upstream source mismatch')


@contextmanager
def verified_atom_reuse(sample, output):
    diffusion = sample.diffusion_module
    encoder = diffusion.atom_attn_enc
    validate_atom_sources(sample, encoder)
    signature = inspect.signature(encoder.get_atom_reps)
    originals = []
    state = None
    report = dict(schema_version=1, mode='atom_native_recomputation_every_call',
                  scopes=[], performance_evidence=False)

    def replace(obj, name, factory):
        originals.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, factory(getattr(obj, name)))

    def wrap_rollout(original):
        def call(*args, **kwargs):
            nonlocal state
            if state is not None:
                raise RuntimeError('Nested atom reuse rollout')
            if (encoder.training or torch.is_grad_enabled()
                    or not torch.are_deterministic_algorithms_enabled()
                    or torch.is_deterministic_algorithms_warn_only_enabled()):
                raise RuntimeError('Atom reuse requires strict deterministic inference')
            record = dict(calls=[], completed=False)
            report['scopes'].append(record)
            state = dict(record=record, retained=None, pending=None)
            try:
                result = original(*args, **kwargs)
                if state['pending'] is not None:
                    raise RuntimeError('Atom conditioning consumer was not observed')
                record['completed'] = True
                return result
            finally:
                state = None
        return call

    def wrap_atoms(original):
        def call(*args, **kwargs):
            if state is None:
                return original(*args, **kwargs)
            if state['pending'] is not None:
                raise RuntimeError('Overlapping atom conditioning consumers')
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            rl = bound.arguments['rl']
            if type(rl) is not torch.Tensor or rl.dtype != torch.float32:
                raise RuntimeError('Atom reuse diagnostic requires FP32 noisy coordinates')
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state(rl.device) if rl.is_cuda else None
            record = dict(index=len(state['record']['calls']), completed=False)
            state['record']['calls'].append(record)
            ql, cl, plm = original(*args, **kwargs)
            first = state['retained'] is None
            if first:
                state['retained'] = (cl, plm)
                state['record']['retained_bytes'] = sum(t.numel() * t.element_size()
                                                       for t in (cl, plm))
            saved_cl, saved_plm = state['retained']
            # The only rl-dependent expression in the audited get_atom_reps.
            candidate_ql = saved_cl + encoder.noisy_position_embedder.linear_r(rl)
            record.update(ql_equal=same_bits(candidate_ql, ql),
                          cl_equal=same_bits(saved_cl, cl),
                          plm_equal=same_bits(saved_plm, plm),
                          cpu_rng_unchanged=torch.equal(cpu_rng, torch.get_rng_state()),
                          gpu_rng_unchanged=(gpu_rng is None or torch.equal(
                              gpu_rng, torch.cuda.get_rng_state(rl.device))))
            if not all(record[k] for k in ('ql_equal', 'cl_equal', 'plm_equal',
                                           'cpu_rng_unchanged', 'gpu_rng_unchanged')):
                raise RuntimeError('Atom reuse differs from native recomputation')
            # First return aliases retained tensors; later native outputs are
            # independent references because consumers receive the cached pair.
            state['pending'] = ((cl.clone(), plm.clone()) if first else (cl, plm), record)
            return candidate_ql, saved_cl, saved_plm
        return call

    def wrap_consumer(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if state is not None:
                if state['pending'] is None:
                    raise RuntimeError('Missing atom conditioning observation')
                references, record = state['pending']
                record['consumer_unchanged'] = all(same_bits(a, b) for a, b in
                                                  zip(state['retained'], references))
                if not record['consumer_unchanged']:
                    raise RuntimeError('Consumer mutated retained atom conditioning')
                record['completed'] = True
                state['pending'] = None
            return result
        return call

    try:
        replace(sample, '_sample_rollout', wrap_rollout)
        replace(encoder, 'get_atom_reps', wrap_atoms)
        replace(diffusion, 'forward', wrap_consumer)
        yield report
    finally:
        for obj, name, had_local, local in reversed(originals):
            if had_local:
                setattr(obj, name, local)
            else:
                delattr(obj, name)
        Path(output, 'atom-reuse-check.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n')
