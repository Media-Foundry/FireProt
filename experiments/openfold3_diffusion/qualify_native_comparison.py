"""Run and verify both frozen routes before any direct-native paired timing."""
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .prepare_native_comparison import read, pin, verify, write
else:
    from prepare_native_comparison import read, pin, verify, write


OPTIMIZED_RECEIPTS = ("operator-runtime.json", "global-softmax.json", "qkv-fusion.json",
                      "gate-fusion.json", "normalization.json", "template-reuse.json",
                      "isolated-data.json", "triangle-key-copy.json", "conditioning-cache.json",
                      "atom-cache.json")


def verify_sealed_outputs(output, required):
    accepted = read(output / "accepted.json")
    outputs = {item["path"]: item for item in accepted["outputs"]}
    if not set(required).issubset(outputs):
        raise ValueError("Required route evidence is absent from the sealed outputs")
    for item in accepted["metadata"] + accepted["outputs"]:
        path = output / item["path"]
        if Path(item["path"]).is_absolute() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Output receipt escapes the prediction directory")
        verify(dict(path=str(path), sha256=item["sha256"]))


def qualify(root):
    root = Path(root).absolute()
    prep = read(root / "preparation.json")
    for reference in prep["files"] + [prep[key] for key in
                                      ("cli", "paired_qualification", "source_preparation",
                                       "reference_npz", "reference_cif", "router", "reference_worker")]:
        verify(reference)
    cli = prep["cli"]["path"]
    records, runners = {}, {}
    for arm, backend in (("baseline", "native"), ("candidate", "hip_fused")):
        spec = read(root / (arm + "-qualification.json"))
        request = read(verify(spec["request"]))
        verify(spec["profile"])
        if request["backend"] != backend or request["hip_device"] != 7:
            raise ValueError("Unexpected qualification backend or device")
        measurement = root / (arm + "-qualification-measurement")
        subprocess.run([cli, "measure-inference", str(root / (arm + "-qualification.json")),
                        str(measurement)], check=True)
        output = Path(request["output_dir"])
        required = ["openfold3-route.json", "runner.json"]
        if backend == "hip_fused":
            required += list(OPTIMIZED_RECEIPTS)
        verify_sealed_outputs(output, required)
        route = read(output / "openfold3-route.json")
        declared = prep["routes"]["workers"][backend]
        selected = root / "python/fireprot_workers" / (declared["module"] + ".py")
        if (route["backend"] != backend or route["selected_worker"] != dict(
                module=declared["module"], path=str(selected), sha256=declared["sha256"])
                or route["reference_worker_sha256"] != prep["reference_worker"]["sha256"]):
            raise ValueError("Executed route differs from the frozen manifest")
        arrays = json.loads(subprocess.check_output([cli, "compare-npz", prep["reference_npz"]["path"],
                                                    str(output / "sample-0.npz")]))
        if len(arrays) != 11 or not all(x["passed"] and x["bitwise_equal"] for x in arrays.values()):
            raise ValueError("Routed model failed the exact 11-array gate")
        predictions = read(output / "predictions.json")
        cif = output / predictions["predictions"][0]["structure"]
        if cif.read_bytes() != Path(prep["reference_cif"]["path"]).read_bytes():
            raise ValueError("Routed model failed complete CIF byte equality")
        if backend == "native":
            if any((output / name).exists() for name in OPTIMIZED_RECEIPTS):
                raise ValueError("Native route emitted an optimized execution receipt")
        else:
            global_attention = read(output / "global-softmax.json")
            if (global_attention["calls"] != 4800 or global_attention["probability_calls"] != 4800
                    or global_attention["hip_bias_copies"] != 4800 or global_attention["groups_launched"] != 19200
                    or global_attention["bias_copies"] != 0 or global_attention["score_layout_fallbacks"] != 0
                    or global_attention["pack_tile_rows"] != 64):
                raise ValueError("Optimized global attention coverage changed")
            qkv = read(output / "qkv-fusion.json")
            if qkv["calls_by_batch"] != {"128": 5064} or qkv["fallback_by_batch"]:
                raise ValueError("Optimized QKV coverage changed")
            if read(output / "normalization.json")["counts"] != dict(
                    accepted=11202, fallback=7846, contiguous_copies=4730):
                raise ValueError("Optimized normalization coverage changed")
        runners[arm] = read(output / "runner.json")
        records[arm] = dict(backend=backend, route=route, arrays=arrays, full_cif_exact=True,
                            output_dir=str(output), npz=pin(output / "sample-0.npz"), cif=pin(cif),
                            accepted=pin(output / "accepted.json"), invocation=pin(measurement / "invocation.json"),
                            e2e_seconds=read(measurement / "invocation.json")["elapsed_seconds"],
                            model_seconds=predictions["model_seconds"])
    optimized_runner = dict(runners["candidate"])
    if optimized_runner.pop("data_module_args", None) != dict(num_workers=0):
        raise ValueError("Expected the qualified candidate data preparation")
    if "data_module_args" in runners["baseline"] or optimized_runner != runners["baseline"]:
        raise ValueError("Native defaults or common model runner settings changed")
    write(root / "route-qualification.json", dict(schema_version=1, correctness_passed=True,
          all11_exact=True, full_cif_exact=True, records=records, preparation=pin(root / "preparation.json"),
          scope="Both routed models qualified; individual timings are not a paired speedup"))


if __name__ == "__main__":
    qualify(Path(sys.argv[1]))
