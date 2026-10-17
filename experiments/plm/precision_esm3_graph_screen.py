"""Precision diagnostic: exact sequence-only HIP Graph replay, not an E2E route."""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("profile")
    parser.add_argument("output")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    profile = json.loads(Path(args.profile).read_text())
    os.environ.update(profile["environment"])
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    sys.path.insert(0, request["source"]["root"])
    import torch
    from fireprot.adapters.plm.runtime import optimized_scope
    from fireprot_workers.plm import load_model
    from fireprot_workers.plm_batch import prepare_forward

    output = Path(args.output)
    output.mkdir(exist_ok=False, parents=True)
    source = Path(request["input"]["path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == request["input"]["sha256"]
    sequences = [item["sequence"] for item in json.loads(source.read_text())["sequences"]]
    assert len(sequences) == 4 and all(len(s) == 1000 for s in sequences)
    torch.set_num_threads(8)
    torch.manual_seed(request["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.cuda.tunable.is_enabled()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1100"
    report = dict(schema_version=1, model="esm3-open", seed=request["seed"],
                  precision="native BF16", torch=str(torch.__version__), hip=torch.version.hip,
                  device="HIP0 / 0000:53:00.0 / gfx1100", length=1000,
                  boundary="resident model; replay includes token copy; excludes loading/export; not E2E",
                  cache_state="warmed operators; filesystem and GPU caches not flushed",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256=request["input"]["sha256"], cases=[])

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def exact(expected, actual):
        return {key: torch.equal(value.contiguous().view(torch.uint8),
                                 actual[key].contiguous().view(torch.uint8))
                for key, value in expected.items()}

    def timed(callback):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = callback()
        torch.cuda.synchronize()
        return result, time.perf_counter() - begin

    @contextmanager
    def capture_scope(model, size):
        # Only sequence_tokens are supplied to the original ESM3 forward.
        # The original constructor consequently always creates NaN coordinates
        # and an all-false mask. Validate that property in an eager prepass;
        # eliminate only the host bool(mask.any()) synchronization for capture.
        changes = []
        try:
            for block in model.transformer.blocks:
                if not block.use_geom_attn:
                    continue
                module = block.geom_attn
                assert module.mask_and_zero_frameless and not module.training
                native = module.forward

                def geometry(s, affine, affine_mask, sequence_id, chain_id, _module=module):
                    assert s.shape == (size, 1002, 1536) and s.dtype == torch.bfloat16
                    assert s.is_contiguous() and not torch.is_grad_enabled()
                    if not torch.cuda.is_current_stream_capturing():
                        assert affine_mask.dtype == torch.bool and not bool(affine_mask.any())
                    zero = torch.zeros((*s.shape[:-1], _module.out_proj.in_features),
                                       device=s.device, dtype=s.dtype)
                    return _module.out_proj(zero)

                changes.append((module, native))
                module.forward = geometry
            assert len(changes) == 1
            yield
        finally:
            for module, native in reversed(changes):
                module.forward = native

    report["isolation_before"] = json.loads(subprocess.check_output(
        [profile["amd_smi"], "snapshot-v1", "--json"], text=True))
    save()
    with torch.inference_mode():
        model, _, _ = load_model(request, sequences[0])
        for size in (1, 4):
            case = dict(batch_size=size)
            report["cases"].append(case)
            save()
            forward = prepare_forward(model, "esm3-open", sequences[:size])
            tokens = inspect.getclosurevars(forward).nonlocals["tokens"]
            original_tokens = tokens.clone()

            def eager():
                with optimized_scope(model, request, 1000, batch_size=size):
                    return forward()

            reference, first_seconds = timed(eager)
            reference = {k: v.clone() for k, v in reference.items()}
            _, case["warm_eager_seconds"] = timed(eager)
            case["first_eager_seconds"] = first_seconds
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with capture_scope(model, size):
                begin = time.perf_counter()
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        forward()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                case["side_stream_warmup_seconds"] = time.perf_counter() - begin
                graph = torch.cuda.CUDAGraph()
                case["allocated_before_capture"] = torch.cuda.memory_allocated()
                begin = time.perf_counter()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        static = forward()
                    torch.cuda.synchronize()
                except Exception as error:
                    case["capture_error"] = repr(error)
                    save()
                    raise
                case["capture_seconds"] = time.perf_counter() - begin
                case["allocated_after_capture"] = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            case["initial_exact"] = exact(reference, static)
            assert all(case["initial_exact"].values())
            alternate_sequences = sequences[1:2] if size == 1 else list(reversed(sequences))
            alternate_forward = prepare_forward(model, "esm3-open", alternate_sequences)
            with optimized_scope(model, request, 1000, batch_size=size):
                alternate = alternate_forward()
            assert not torch.equal(alternate["tokens"], reference["tokens"])
            tokens.copy_(alternate["tokens"])
            graph.replay()
            torch.cuda.synchronize()
            case["changed_tokens_exact"] = exact(alternate, static)
            assert all(case["changed_tokens_exact"].values())
            tokens.copy_(original_tokens)
            graph.replay()
            torch.cuda.synchronize()
            assert all(exact(reference, static).values())

            def replay():
                tokens.copy_(original_tokens)
                graph.replay()
                return static

            samples = {"eager": [], "graph": []}
            case["exact_runs"] = []
            for order in (("eager", "graph"), ("graph", "eager")) * 4:
                for arm in order:
                    actual, seconds = timed(eager if arm == "eager" else replay)
                    samples[arm].append(seconds)
                    check = exact(reference, actual)
                    case["exact_runs"].append(dict(arm=arm, exact=check))
                    assert all(check.values())
            case["samples_seconds"] = samples
            case["medians_seconds"] = {k: statistics.median(v) for k, v in samples.items()}
            case["speedup"] = case["medians_seconds"]["eager"] / case["medians_seconds"]["graph"]
            saving = case["medians_seconds"]["eager"] - case["medians_seconds"]["graph"]
            setup = case["side_stream_warmup_seconds"] + case["capture_seconds"]
            case["additional_setup_seconds"] = setup
            case["estimated_replays_to_amortize_setup"] = math.ceil(setup / saving) if saving > 0 else None
            save()
            print(json.dumps({k: v for k, v in case.items() if k not in ("exact_runs", "samples_seconds")}), flush=True)
            del actual, static, graph, reference, alternate, alternate_forward, forward
            torch.cuda.empty_cache()
    report["isolation_after"] = json.loads(subprocess.check_output(
        [profile["amd_smi"], "snapshot-v1", "--json"], text=True))
    save()


if __name__ == "__main__":
    main()
