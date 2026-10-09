"""Recheck current output bytes and route receipts after direct-native pairs."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

if __package__:
    from .prepare_native_comparison import read, pin, verify, write
    from .qualify_native_comparison import verify_sealed_outputs, OPTIMIZED_RECEIPTS
else:
    from prepare_native_comparison import read, pin, verify, write
    from qualify_native_comparison import verify_sealed_outputs, OPTIMIZED_RECEIPTS


def collect(root):
    root = Path(root).absolute()
    paired = root / "paired"
    prep = read(root / "preparation.json")
    qualification = read(root / "route-qualification.json")
    if not (qualification["correctness_passed"] and qualification["all11_exact"]
            and qualification["full_cif_exact"]):
        raise ValueError("Both routed full-model qualifications are required")
    verify(qualification["preparation"])
    for ref in prep["files"] + [prep[key] for key in
                               ("cli", "paired_qualification", "source_preparation",
                                "reference_npz", "reference_cif", "router", "reference_worker")]:
        verify(ref)
    benchmark = read(paired / "benchmark.json")
    if benchmark["phase"] != "succeeded" or len(benchmark["measurements"]) != 6:
        raise ValueError("Six successful paired measurements are required")

    def resolve(ref):
        path = Path(ref["path"])
        if not path.is_absolute():
            path = paired / path
        return verify(dict(path=str(path), sha256=ref["sha256"]))

    comparison_path = resolve(benchmark["comparison"])
    comparison = read(comparison_path)
    if not comparison["correctness_passed"] or len(comparison["pairs"]) != 3:
        raise ValueError("Paired numerical comparison failed")
    expected_slots = {(a, i) for a in ("baseline", "candidate") for i in range(3)}
    if {(x["slot"]["arm"], x["slot"]["pair"]) for x in benchmark["measurements"]} != expected_slots:
        raise ValueError("Missing or duplicate paired slots")
    records = []
    for measurement in benchmark["measurements"]:
        arm, pair = measurement["slot"]["arm"], measurement["slot"]["pair"]
        accepted = resolve(measurement["accepted"])
        output = accepted.parent
        invocation_path = resolve(measurement["invocation"])
        invocation = read(invocation_path)
        if invocation["exit_code"] != 0 or resolve(invocation["accepted"]) != accepted:
            raise ValueError("Invocation does not identify the successful output")
        required = ["openfold3-route.json", "runner.json"]
        if arm == "candidate":
            required += list(OPTIMIZED_RECEIPTS)
        verify_sealed_outputs(output, required)
        anchor = qualification["records"][arm]
        route = read(output / "openfold3-route.json")
        if route != anchor["route"]:
            raise ValueError("Paired run changed its selected model worker")
        anchor_output = Path(anchor["output_dir"])
        verify(anchor["accepted"])
        verify_sealed_outputs(anchor_output, required)
        if read(output / "runner.json") != read(anchor_output / "runner.json"):
            raise ValueError("Paired model runner differs from the qualified route")
        if arm == "baseline":
            if any((output / name).exists() for name in OPTIMIZED_RECEIPTS):
                raise ValueError("Native baseline emitted optimized execution evidence")
        else:
            # Dispatch/copy counts and selected libraries must equal the exact model gate.
            for name in OPTIMIZED_RECEIPTS:
                current, reference = read(output / name), read(anchor_output / name)
                if name == "operator-runtime.json":
                    if current["launches_submitted"] != reference["launches_submitted"]:
                        raise ValueError("Shared native launch counter changed")
                elif name in ("global-softmax.json", "qkv-fusion.json", "gate-fusion.json", "normalization.json"):
                    if current != reference:
                        raise ValueError(f"Optimized execution changed: {name}")
        arrays = json.loads(subprocess.check_output([prep["cli"]["path"], "compare-npz",
                             prep["reference_npz"]["path"], str(output / "sample-0.npz")]))
        if len(arrays) != 11 or not all(v["passed"] and v["bitwise_equal"] for v in arrays.values()):
            raise ValueError("Current paired arrays are not exactly qualified")
        prediction = read(output / "predictions.json")
        cif = output / prediction["predictions"][0]["structure"]
        if cif.read_bytes() != Path(prep["reference_cif"]["path"]).read_bytes():
            raise ValueError("Current paired CIF bytes differ")
        records.append(dict(arm=arm, pair=pair, route=route, arrays=arrays, full_cif_exact=True,
                            e2e_seconds=invocation["elapsed_seconds"], model_seconds=prediction["model_seconds"],
                            accepted=pin(accepted), invocation=pin(invocation_path),
                            npz=pin(output / "sample-0.npz"), cif=pin(cif)))
    performance = comparison["performance"]
    model_medians = {}
    for arm in ("baseline", "candidate"):
        observed = [x for x in records if x["arm"] == arm]
        if statistics.median(x["e2e_seconds"] for x in observed) != performance[arm]["median"]:
            raise ValueError("Published median differs from raw measurements")
        model_medians[arm] = statistics.median(x["model_seconds"] for x in observed)
    write(root / "collected.json", dict(schema_version=1, correctness_passed=True, all11_exact=True,
          full_cif_exact=True, scope=prep["scope"], performance=performance, model_medians=model_medians,
          model_speedup=model_medians["baseline"] / model_medians["candidate"], records=records,
          comparison=pin(comparison_path), preparation=pin(root / "preparation.json"),
          qualification=pin(root / "route-qualification.json"),
          limits=["Direct native Torch under the common FireProt launcher, not an unwrapped upstream CLI.",
                  "Strict FP32 N1536 R3/four trunk passes D200/S1 seed101; native data defaults retained.",
                  "Three AB/BA/AB pairs, fresh processes, unflushed caches, HIP7 only; no observations discarded."]))


if __name__ == "__main__":
    collect(Path(sys.argv[1]))
