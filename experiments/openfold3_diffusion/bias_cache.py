"""Experimental rollout-local reuse of global diffusion pair projections.

Only the pinned immutable FP32 inference workload is eligible. Verification mode
recomputes every native value and checks it before and after its consumers. The
timing mode relies on those audited invariants; identity guards do not detect
in-place mutations of inference tensors or parameters.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

import torch


def validate_sources(sample):
    transformer = sample.diffusion_module.diffusion_transformer
    for obj, expected in (
        (sample, "3b5ed9fb090d53df2b5ad16be517cfdca7f5290a6bb8274e71408a3ff465155d"),
        (transformer, "a2f1abff3c59a141311f1408b084f96235332067a95a35e5133a909aff9796e6"),
        (transformer.blocks[0].attention_pair_bias,
         "a7622c8a315e917480586047af7fb35ed2283ade0a16bcf8b68f8f5f22f5cdee"),
    ):
        if hashlib.sha256(Path(inspect.getfile(type(obj))).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Diffusion bias cache source mismatch")
    if len(transformer.blocks) != 24:
        raise RuntimeError("Expected 24 global diffusion blocks")
    return transformer


def tensor_key(value):
    return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), value.dtype, value.device)


def eligible(value):
    return (type(value) is torch.Tensor and value.is_cuda and value.dtype == torch.float32
            and not value.requires_grad and not value.is_neg() and not value.is_conj()
            and tuple(value.shape) == (1, 1, 1536, 1536, 128))


def exact(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def pack_projection(value):
    """Keep logical [..., N, N, H] values, store the eventual bias head first."""
    axes = list(range(value.ndim - 3))
    bias = value.permute(*axes, -1, -3, -2).contiguous()
    return bias.permute(*axes, -2, -1, -3)


@contextmanager
def global_bias_cache(sample, output, verify=False, pack_bias=False):
    transformer = validate_sources(sample)
    diffusion = sample.diffusion_module
    targets = [("normalization", transformer.layer_norm_z)] + [
        (f"projection.{i}", block.attention_pair_bias.linear_z)
        for i, block in enumerate(transformer.blocks)
    ]
    report = dict(schema_version=1, verify=verify, pack_bias=pack_bias, scopes=[],
                  scope="Global diffusion transformer only; " + (
                      "head-contiguous projected bias" if pack_bias else "original output layouts"),
                  limits=["Timing mode requires source-pinned immutable inputs and parameters.",
                          "Identity checks do not detect in-place inference tensor mutation.",
                          "Verification mode is diagnostic and cannot establish performance."])
    state = None
    saved = []

    def replace(obj, name, function):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, function)

    rollout = sample._sample_rollout
    denoise = diffusion.forward

    def run_rollout(*args, **kwargs):
        nonlocal state
        if state is not None:
            raise RuntimeError("Nested global bias cache rollout")
        if (transformer.training or not torch.is_inference_mode_enabled()
                or torch.is_grad_enabled() or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()):
            raise RuntimeError("Global bias cache requires strict deterministic inference")
        record = dict(completed=False, denoiser_calls=0, misses=0, hits=0,
                      fallbacks=0, invalidations=0, native_checks=0,
                      consumer_checks=0, retained_bytes=0, entries={})
        report["scopes"].append(record)
        state = dict(record=record, cache={}, expected={}, active=False)
        try:
            result = rollout(*args, **kwargs)
            record["completed"] = True
            return result
        finally:
            state = None

    def run_denoise(*args, **kwargs):
        if state is None:
            return denoise(*args, **kwargs)
        if state["active"]:
            raise RuntimeError("Nested denoiser call")
        state["active"] = True
        state["expected"].clear()
        record = state["record"]
        record["denoiser_calls"] += 1
        try:
            result = denoise(*args, **kwargs)
            for name, expected in state["expected"].items():
                if not exact(state["cache"][name]["value"], expected):
                    raise RuntimeError(f"Cached bias was modified by a consumer: {name}")
                record["consumer_checks"] += 1
            return result
        finally:
            state["expected"].clear()
            state["active"] = False

    def wrap(name, module, original):
        def forward(value, *args, **kwargs):
            if state is None or not state["active"]:
                return original(value, *args, **kwargs)
            record, cache = state["record"], state["cache"]
            if args or kwargs or not eligible(value):
                cache.pop(name, None)
                record["fallbacks"] += 1
                return original(value, *args, **kwargs)
            parameters = tuple(module.parameters())
            key = (tensor_key(value), tuple(tensor_key(p) for p in parameters))
            entry = cache.get(name)
            if entry is not None and entry["key"] != key:
                record["invalidations"] += 1
                cache.pop(name)
                entry = None
            if entry is None:
                native = original(value)
                retained = pack_projection(native) if pack_bias and name.startswith("projection.") else native
                entry = dict(key=key, value=retained, owners=(value, parameters))
                cache[name] = entry
                record["misses"] += 1
                if verify:
                    state["expected"][name] = native.clone()
            else:
                record["hits"] += 1
                if verify:
                    native = original(value)
                    if not exact(native, entry["value"]):
                        raise RuntimeError(f"Native bias differs from retained value: {name}")
                    state["expected"][name] = native
                    record["native_checks"] += 1
            record["entries"][name] = dict(shape=list(entry["value"].shape),
                strides=list(entry["value"].stride()),
                bytes=entry["value"].numel() * entry["value"].element_size())
            record["retained_bytes"] = sum(e["value"].numel() * e["value"].element_size()
                                            for e in cache.values())
            return entry["value"]
        return forward

    try:
        replace(sample, "_sample_rollout", run_rollout)
        replace(diffusion, "forward", run_denoise)
        for name, module in targets:
            replace(module, "forward", wrap(name, module, module.forward))
        yield report
    finally:
        for obj, name, existed, old in reversed(saved):
            if existed:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        Path(output, "global-bias-cache.json").write_text(json.dumps(report, indent=2) + "\n")
