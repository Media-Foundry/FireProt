"""Read-only conditioning observations in an isolated OpenFold3 worker.

This diagnostic performs CPU copies and retains sampled pair outputs until their
consumers finish. Its timings and memory use are not performance measurements.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path

import torch


def _bytes(tensor):
    return tensor.detach().to(device="cpu", copy=True).contiguous().view(torch.uint8)


def _metadata(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), device=str(tensor.device))


@contextmanager
def conditioning_probe(sample_module, output, sampled_calls=(0, 1, 99, 199)):
    """Observe each sampling invocation separately, restoring instance methods.

    Arguments are a SampleDiffusion instance and the Rust-owned output directory.
    No replacement tensor is returned and no caching optimization is enabled.
    """
    diffusion = sample_module.diffusion_module
    conditioning = diffusion.diffusion_conditioning
    sampled_calls = frozenset(sampled_calls)
    report = dict(schema_version=1, scopes=[], sampled_calls=sorted(sampled_calls),
                  limits=["Only selected calls receive full bitwise observation.",
                          "CPU copies and retained outputs perturb time and memory.",
                          "Observed output equality does not prove input immutability.",
                          "No caching or performance result is established."])
    originals = []
    scope = None
    active_record = None
    pending = None
    reference = None
    signature = inspect.signature(conditioning.forward)

    def replace(obj, name, factory):
        original = getattr(obj, name)
        originals.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, factory(original))

    def wrap_sample(original):
        def call(*args, **kwargs):
            nonlocal scope, reference, pending
            if scope is not None:
                raise RuntimeError("Nested sampling is outside this diagnostic's scope")
            scope = dict(calls=[], completed=False)
            report["scopes"].append(scope)
            reference = None
            try:
                result = original(*args, **kwargs)
                scope["completed"] = True
                return result
            finally:
                reference = pending = scope = None
        return call

    def wrap_conditioning(original):
        def call(*args, **kwargs):
            nonlocal active_record, reference, pending
            if scope is None:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = bound.arguments
            index = len(scope["calls"])
            record = dict(index=index, completed=False,
                          use_conditioning=inputs["use_conditioning"],
                          requested_chunk=inputs["chunk_size"], chunks=[],
                          inputs={k: _metadata(inputs[k]) for k in
                                  ("si_input", "si_trunk", "zij_trunk", "t")})
            scope["calls"].append(record)
            selected = index in sampled_calls
            device = inputs["zij_trunk"].device
            cpu_rng = torch.get_rng_state() if selected else None
            gpu_rng = (torch.cuda.get_rng_state(device) if selected and
                       device.type == "cuda" else None)
            active_record = record
            try:
                result = original(*args, **kwargs)
                record["completed"] = True
                if selected:
                    record["noise_level"] = inputs["t"].detach().cpu().tolist()
                    pair = result[1]
                    snapshot = _bytes(pair)
                    record["pair"] = _metadata(pair)
                    if reference is None:
                        reference = snapshot
                        record["reference_call"] = True
                    else:
                        record["equal_to_first_sampled_pair"] = torch.equal(reference, snapshot)
                    record["cpu_rng_unchanged"] = torch.equal(cpu_rng, torch.get_rng_state())
                    if gpu_rng is not None:
                        record["gpu_rng_unchanged"] = torch.equal(
                            gpu_rng, torch.cuda.get_rng_state(device))
                    # Release at the enclosing diffusion call's exit. Retaining
                    # the reference deliberately makes this a diagnostic run.
                    pending = (pair, snapshot, record)
                return result
            finally:
                active_record = None
        return call

    def wrap_chunks(original):
        chunk_signature = inspect.signature(original)
        def call(*args, **kwargs):
            if active_record is not None:
                bound = chunk_signature.bind(*args, **kwargs)
                bound.apply_defaults()
                active_record["chunks"].append(bound.arguments["chunk_size"])
            return original(*args, **kwargs)
        return call

    def wrap_diffusion(original):
        def call(*args, **kwargs):
            nonlocal pending
            try:
                result = original(*args, **kwargs)
                if pending is not None:
                    pair, snapshot, record = pending
                    record["pair_unchanged_after_consumers"] = torch.equal(snapshot, _bytes(pair))
                return result
            finally:
                pending = None
        return call

    try:
        replace(sample_module, "forward", wrap_sample)
        replace(conditioning, "forward", wrap_conditioning)
        replace(conditioning, "_forward", wrap_chunks)
        replace(diffusion, "forward", wrap_diffusion)
        yield report
    finally:
        for obj, name, had_local, local in reversed(originals):
            if had_local:
                setattr(obj, name, local)
            else:
                delattr(obj, name)
        Path(output, "conditioning-audit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
