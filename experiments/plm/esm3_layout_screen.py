"""Screen alternate BF16 weight storage using actual ESM3 activations."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys


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
    output.mkdir(parents=True, exist_ok=False)
    source = Path(request["input"]["path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == request["input"]["sha256"]
    sequences = [item["sequence"] for item in json.loads(source.read_text())["sequences"]]
    torch.set_num_threads(8)
    torch.manual_seed(request["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.cuda.tunable.is_enabled()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1100"
    native = torch.nn.functional.linear
    shapes = {(1536, 1536), (4608, 1536), (8192, 1536), (1536, 4096)}
    rows = collections.defaultdict(lambda: dict(calls=0, exact_calls=0, unequal_elements=0, max_abs=0.0))
    retained = {}

    def screened(x, weight, bias=None):
        reference = native(x, weight, bias)
        if tuple(weight.shape) in shapes and bias is None and x.shape[:2] == (4, 1002):
            transposed = weight.t().contiguous().t()
            candidate = native(x, transposed, bias)
            key = str(tuple(weight.shape))
            row = rows[key]
            same = torch.equal(reference.view(torch.uint8), candidate.view(torch.uint8))
            row["calls"] += 1
            row["exact_calls"] += int(same)
            row["unequal_elements"] += int(torch.count_nonzero(reference != candidate))
            row["max_abs"] = max(row["max_abs"], float((reference.float() - candidate.float()).abs().max()))
            if key not in retained:
                retained[key] = (x.clone(), weight, transposed)
        return reference

    with torch.inference_mode():
        model, _, _ = load_model(request, sequences[0])
        forward = prepare_forward(model, "esm3-open", sequences)
        torch.nn.functional.linear = screened
        try:
            with optimized_scope(model, request, 1000, batch_size=4):
                forward()
        finally:
            torch.nn.functional.linear = native
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for key, (x, weight, transposed) in retained.items():
                def measure(w):
                    for _ in range(3):
                        native(x, w)
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(10):
                        native(x, w)
                    b.record()
                    b.synchronize()
                    return a.elapsed_time(b) / 10
                samples = {"native": [], "transposed": []}
                for order in (("native", "transposed"), ("transposed", "native")) * 3:
                    for arm in order:
                        samples[arm].append(measure(weight if arm == "native" else transposed))
                row = rows[key]
                row["samples_ms"] = samples
                row["medians_ms"] = {k: statistics.median(v) for k, v in samples.items()}
                row["speedup"] = row["medians_ms"]["native"] / row["medians_ms"]["transposed"]
                row["native_weight_stride"] = list(weight.stride())
                row["transposed_weight_stride"] = list(transposed.stride())
    report = dict(schema_version=1, model="esm3-open", batch_size=4, length=1000,
                  seed=request["seed"], precision="native BF16", torch=str(torch.__version__),
                  hip=torch.version.hip, architecture="gfx1100", shapes=dict(rows),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256=request["input"]["sha256"],
                  boundary="isolated warm GEMMs; weight conversion excluded; not model/E2E timing")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in row.items() if k != "samples_ms"}
                      for key, row in rows.items()}), flush=True)


if __name__ == "__main__":
    main()
