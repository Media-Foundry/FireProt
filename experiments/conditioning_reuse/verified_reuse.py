"""Experimental conditioning reuse with mandatory native recomputation.

This is a correctness diagnostic, never a performance implementation. Every
conditioning call still executes upstream and verifies the proposed return values.
"""
# The single-branch arithmetic follows OpenFold3 DiffusionConditioning:
# Copyright 2026 AlQuraishi Laboratory, licensed under Apache-2.0.
# https://www.apache.org/licenses/LICENSE-2.0
from contextlib import contextmanager
import inspect
import json
from pathlib import Path

import torch


def single_conditioning(module, inputs, chunk_size):
    """Reproduce the audited native single branch without pair computations."""
    si_trunk = inputs["si_trunk"]
    if not inputs["use_conditioning"]:
        si_trunk = si_trunk * 0
    si = torch.cat([si_trunk, inputs["si_input"]], dim=-1)
    si = module.linear_s(module.layer_norm_s(si))
    n = 0.25 * torch.log(inputs["t"] / module.sigma_data)
    n = module.fourier_emb(n.unsqueeze(-1))
    si = si + module.linear_n(module.layer_norm_n(n)).unsqueeze(-2)
    for layer in module.transition_s:
        si = si + layer(si, mask=inputs["batch"]["token_mask"],
                        chunk_size=chunk_size)
    return si


def same_bits(a, b):
    return (a.shape == b.shape and a.stride() == b.stride() and
            a.dtype == b.dtype and a.device == b.device and
            torch.equal(a.contiguous().view(torch.uint8),
                        b.contiguous().view(torch.uint8)))


@contextmanager
def verified_reuse(sample_module, output):
    """Scope retained pairs to one rollout, checking every call and consumer.

    Supports the audited inference module only. No unchecked reuse mode exists.
    The worker must pin the upstream source and enable strict determinism.
    """
    diffusion = sample_module.diffusion_module
    conditioning = diffusion.diffusion_conditioning
    signature = inspect.signature(conditioning.forward)
    original_methods = []
    scope = pair = pending = chunks = None
    report = dict(schema_version=1, mode="native_recomputation_every_call",
                  scopes=[], performance_evidence=False)

    def replace(obj, name, factory):
        original = getattr(obj, name)
        original_methods.append((obj, name, name in obj.__dict__,
                                 obj.__dict__.get(name)))
        setattr(obj, name, factory(original))

    def wrap_rollout(original):
        def call(*args, **kwargs):
            nonlocal scope, pair, pending
            if scope is not None:
                raise RuntimeError("Nested conditioning reuse scope")
            if conditioning.training or torch.is_grad_enabled():
                raise RuntimeError("Conditioning reuse requires inference execution")
            scope = dict(calls=[], completed=False)
            report["scopes"].append(scope)
            try:
                result = original(*args, **kwargs)
                if pending is not None:
                    raise RuntimeError("Conditioning consumer was not observed")
                scope["completed"] = True
                return result
            finally:
                scope = pair = pending = None
        return call

    def wrap_chunks(original):
        chunk_signature = inspect.signature(original)
        def call(*args, **kwargs):
            if chunks is not None:
                bound = chunk_signature.bind(*args, **kwargs)
                bound.apply_defaults()
                chunks.append(bound.arguments["chunk_size"])
            return original(*args, **kwargs)
        return call

    def wrap_conditioning(original):
        def call(*args, **kwargs):
            nonlocal pair, pending, chunks
            if scope is None:
                return original(*args, **kwargs)
            if pending is not None:
                raise RuntimeError("Overlapping conditioning consumers")
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = bound.arguments
            device = inputs["zij_trunk"].device
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            record = dict(index=len(scope["calls"]), completed=False, chunks=[])
            scope["calls"].append(record)
            chunks = record["chunks"]
            try:
                native_single, native_pair = original(*args, **kwargs)
            finally:
                chunks = None
            if not record["chunks"]:
                raise RuntimeError("Missing native conditioning chunk observation")
            if pair is None:
                pair = native_pair
                candidate_single = native_single
                # First native return aliases the retained pair; clone only for
                # the consumer mutation check, then release at diffusion exit.
                pending_pair = native_pair.clone()
                record["reused"] = False
                scope["retained_pair_bytes"] = pair.numel() * pair.element_size()
            else:
                candidate_single = single_conditioning(
                    conditioning, inputs, record["chunks"][-1])
                pending_pair = native_pair
                record["reused"] = True
            record["single_equal"] = same_bits(candidate_single, native_single)
            record["pair_equal"] = same_bits(pair, native_pair)
            record["cpu_rng_unchanged"] = torch.equal(cpu_rng, torch.get_rng_state())
            record["gpu_rng_unchanged"] = (gpu_rng is None or torch.equal(
                gpu_rng, torch.cuda.get_rng_state(device)))
            if not all(record[key] for key in ("single_equal", "pair_equal",
                                               "cpu_rng_unchanged", "gpu_rng_unchanged")):
                raise RuntimeError("Conditioning reuse differs from native recomputation")
            pending = (pending_pair, record)
            record["completed"] = True
            return candidate_single, pair
        return call

    def wrap_diffusion(original):
        def call(*args, **kwargs):
            nonlocal pending
            try:
                result = original(*args, **kwargs)
                if pending is not None:
                    expected, record = pending
                    record["pair_unchanged_after_consumers"] = same_bits(pair, expected)
                    if not record["pair_unchanged_after_consumers"]:
                        raise RuntimeError("Conditioning consumer mutated retained pair")
                return result
            finally:
                pending = None
        return call

    try:
        replace(sample_module, "_sample_rollout", wrap_rollout)
        replace(conditioning, "forward", wrap_conditioning)
        replace(conditioning, "_forward", wrap_chunks)
        replace(diffusion, "forward", wrap_diffusion)
        yield report
    finally:
        for obj, name, had_local, local in reversed(original_methods):
            if had_local:
                setattr(obj, name, local)
            else:
                delattr(obj, name)
        Path(output, "conditioning-reuse.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
