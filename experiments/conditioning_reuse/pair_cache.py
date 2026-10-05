"""Experimental pair reuse for the source-pinned inference rollout.

Inputs and parameters must remain immutable within the audited rollout. Identity
checks below detect replacements, not in-place mutation of inference tensors.
Whole-model bitwise validation is required before any performance claim.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

import torch

if __package__:
    from .verified_reuse import single_conditioning
else:
    from verified_reuse import single_conditioning


def validate_sources(sample, conditioning):
    for obj, expected in (
        (sample, "3b5ed9fb090d53df2b5ad16be517cfdca7f5290a6bb8274e71408a3ff465155d"),
        (conditioning, "5bf5170d836c3b56cf665a9dc68bb935018ef21f779f516bc97ee0be5b1c6305"),
    ):
        source = Path(inspect.getfile(type(obj)))
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Conditioning cache upstream source mismatch")


def eligible(inputs):
    pair = inputs["zij_trunk"]
    return (type(pair) is torch.Tensor and pair.device.type == "cuda" and
            pair.dtype == torch.float32 and pair.ndim == 5 and
            tuple(pair.shape[:2]) == (1, 1) and pair.shape[-1] == 128 and
            pair.shape[-2] == pair.shape[-3] and pair.shape[-2] in (1000, 1536) and
            inputs["use_conditioning"] is True and inputs["chunk_size"] == 1024)


def dependency_key(inputs):
    def identity(value):
        if isinstance(value, torch.Tensor):
            return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                    value.dtype, value.device)
        if isinstance(value, dict):
            return tuple((key, identity(item)) for key, item in sorted(value.items()))
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        return id(value)
    # Time is intentionally excluded. The single branch is recomputed each call.
    return tuple((key, identity(value)) for key, value in sorted(inputs.items()) if key != "t")


@contextmanager
def pair_cache(sample, output):
    conditioning = sample.diffusion_module.diffusion_conditioning
    validate_sources(sample, conditioning)
    signature = inspect.signature(conditioning.forward)
    report = dict(schema_version=1, mode="experimental_pair_cache", scopes=[],
                  limits=["Source-pinned immutable inference inputs are required.",
                          "Identity checks do not detect in-place input/parameter changes."])
    originals = []
    state = None
    chunks = None

    def replace(obj, name, factory):
        originals.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, factory(getattr(obj, name)))

    def wrap_rollout(original):
        def call(*args, **kwargs):
            nonlocal state
            if state is not None:
                raise RuntimeError("Nested pair cache rollout")
            if (conditioning.training or torch.is_grad_enabled() or
                    not torch.are_deterministic_algorithms_enabled() or
                    torch.is_deterministic_algorithms_warn_only_enabled()):
                raise RuntimeError("Pair cache requires strict deterministic inference")
            record = dict(calls=0, hits=0, misses=0, fallbacks=0, completed=False,
                          effective_chunks=[])
            report["scopes"].append(record)
            state = dict(pair=None, key=None, record=record)
            try:
                result = original(*args, **kwargs)
                record["completed"] = True
                return result
            finally:
                state = None
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
            nonlocal chunks
            if state is None:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = bound.arguments
            record = state["record"]
            record["calls"] += 1
            if not eligible(inputs):
                state["pair"] = state["key"] = None
                record["fallbacks"] += 1
                return original(*args, **kwargs)
            key = dependency_key(inputs)
            if state["pair"] is not None and state["key"] == key:
                single = single_conditioning(conditioning, inputs, state["chunk"])
                record["hits"] += 1
                return single, state["pair"]
            # Release old retained storage before native tuning after invalidation.
            state["pair"] = state["key"] = None
            chunks = []
            try:
                single, pair = original(*args, **kwargs)
                if not chunks:
                    raise RuntimeError("Missing actual conditioning chunk")
                state.update(pair=pair, key=key, chunk=chunks[-1])
                record["effective_chunks"].append(chunks[-1])
                record["retained_pair_bytes"] = pair.numel() * pair.element_size()
                record["misses"] += 1
                return single, pair
            finally:
                chunks = None
        return call

    try:
        replace(sample, "_sample_rollout", wrap_rollout)
        replace(conditioning, "forward", wrap_conditioning)
        replace(conditioning, "_forward", wrap_chunks)
        yield report
    finally:
        for obj, name, had_local, local in reversed(originals):
            if had_local:
                setattr(obj, name, local)
            else:
                delattr(obj, name)
        Path(output, "conditioning-cache.json").write_text(json.dumps(report, indent=2) + "\n")
