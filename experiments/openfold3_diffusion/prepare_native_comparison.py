"""Freeze a common native/optimized tree after the HIP packing E2E gate.

This prepares CPU-side artifacts only. Both qualification invocations must pass
before the generated paired specification is executed. External model assets and
qualified kernel locations remain explicit dependencies; they are not copied.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

if __package__:
    from .native_router import REFERENCE_SHA256
else:
    from native_router import REFERENCE_SHA256


FLAGS = ("FIREPROT_CONDITIONING_CACHE", "FIREPROT_ATOM_CACHE", "FIREPROT_SOFTMAX_HEAD16",
         "FIREPROT_QK_WORKSPACE", "FIREPROT_TRIANGLE_MEMORY", "FIREPROT_TEMPLATE_REUSE",
         "FIREPROT_NORM128", "FIREPROT_QKV_INDEX32", "FIREPROT_GLOBAL_SOFTMAX")
COPY_NAMES = ("python", "fireprot", "kernels", "licenses", "LICENSE", "THIRD_PARTY.md",
              "norm.so", "gate.so", "qkv.so", "pack.so", "pack-build.json")


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def verify(reference):
    if pin(reference["path"])["sha256"] != reference["sha256"]:
        raise ValueError(f"Changed prerequisite: {reference['path']}")
    return Path(reference["path"])


def write(path, data):
    with Path(path).open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def verify_copy_inventory(root, references):
    """Reject added or removed source files, not only changed known contents."""
    root = Path(root).absolute()
    expected = {Path(ref["path"]) for ref in references
                if Path(ref["path"]).is_relative_to(root)
                and Path(ref["path"]).relative_to(root).parts[0] in COPY_NAMES}
    observed = set()
    for name in COPY_NAMES:
        path = root / name
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for item in candidates:
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                observed.add(item)
    if observed != expected:
        raise ValueError("Qualified copy inventory changed (added or removed files)")


def comparison_profiles(source, workers):
    profiles = {}
    for arm in ("baseline", "candidate"):
        profile = deepcopy(source)
        profile["workers"] = str(workers)
        for flag in FLAGS:
            if profile["environment"].get(flag) != "1":
                raise ValueError(f"Qualified optimized profile does not enable {flag}")
            profile["environment"][flag] = "0" if arm == "baseline" else "1"
        profiles[arm] = profile
    return profiles


def install_routes(package, reference_worker, router):
    """Keep both worker bodies byte-identical; change only the package entry."""
    reference = pin(reference_worker)
    if reference["sha256"] != REFERENCE_SHA256:
        raise ValueError("Native reference is not the audited unchanged worker")
    package = Path(package)
    original = package / "openfold3.py"
    optimized = package / "openfold3_optimized.py"
    if optimized.exists() or (package / "openfold3_reference.py").exists():
        raise ValueError("Routes already installed")
    optimized.write_bytes(original.read_bytes())
    (package / "openfold3_reference.py").write_bytes(Path(reference_worker).read_bytes())
    original.write_bytes(Path(router).read_bytes())
    routes = dict(schema_version=1, workers={
        "native": dict(module="openfold3_reference", sha256=REFERENCE_SHA256),
        "hip_fused": dict(module="openfold3_optimized", sha256=pin(optimized)["sha256"])})
    write(package / "openfold3_routes.json", routes)
    return routes


def prepare(paired_root, destination, output_root, reference_worker):
    paired_root, destination, output_root = map(lambda p: Path(p).absolute(),
                                               (paired_root, destination, output_root))
    if destination.exists():
        raise ValueError("Destination must be unused")
    collected = read(paired_root / "collected.json")
    if (not collected["correctness_passed"] or len(collected["runs"]) != 6
            or len(collected["performance"]["paired_speedups"]) != 3
            or not all(x > 1 for x in collected["performance"]["paired_speedups"])):
        raise ValueError("HIP packing must pass correctness and every E2E pair first")
    for run in collected["runs"]:
        if (not run["full_cif_bytes_equal"] or len(run["arrays"]) != 11
                or not all(x["passed"] and x["bitwise_equal"] for x in run["arrays"].values())):
            raise ValueError("Incomplete paired numerical qualification")
    source_preparation = verify(collected["preparation"])
    if source_preparation != paired_root / "preparation.json":
        raise ValueError("Paired evidence belongs to a different source tree")
    prep = read(source_preparation)
    for ref in prep["files"] + [prep[key] for key in
                               ("cli", "model_qualification", "reference_npz", "reference_cif")]:
        verify(ref)
    verify_copy_inventory(paired_root, prep["files"])
    if pin(reference_worker)["sha256"] != REFERENCE_SHA256:
        raise ValueError("Native source pin mismatch")
    profiles = comparison_profiles(read(paired_root / "candidate-profile.json"), destination / "python")
    request = read(paired_root / "hip-bias-pack-pairs-n1536-candidate-0.json")
    if (request["hip_device"] != 7 or request["backend"] != "hip_fused" or request["seed"] != 101
            or request["workload"] != dict(model="openfold3", model_name="openbind-0", recycles=3,
                                           diffusion_steps=200, samples=1, precision="float32", deterministic=True)):
        raise ValueError("Unsupported direct-native qualification workload")
    destination.mkdir()
    for name in COPY_NAMES:
        src = paired_root / name
        if src.is_dir():
            shutil.copytree(src, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, destination / name)
    package = destination / "python/fireprot_workers"
    router = Path(__file__).with_name("native_router.py")
    routes = install_routes(package, reference_worker, router)
    transport = package / "transport.py"
    text = transport.read_text()
    needle = "'global-softmax.json',"
    if text.count(needle) != 1:
        raise ValueError("Unexpected qualified transport layout")
    text = text.replace(needle, "'openfold3-route.json', 'runner.json', 'isolated-data.json', "
                        "'triangle-key-copy.json', " + needle)
    compile(text, str(transport), "exec")
    transport.write_text(text)
    profile_pins = {}
    for arm, profile in profiles.items():
        path = destination / (arm + "-profile.json")
        write(path, profile)
        profile_pins[arm] = pin(path)

    def invocation(arm, suffix):
        q = deepcopy(request)
        q.update(backend="native" if arm == "baseline" else "hip_fused",
                 run_id=f"{destination.name}-{arm}-{suffix}")
        q["output_dir"] = str(output_root / q["run_id"])
        if Path(q["output_dir"]).exists():
            raise ValueError("Prediction destination is already used")
        path = destination / (q["run_id"] + ".json")
        write(path, q)
        subprocess.run([prep["cli"]["path"], "inspect", str(path)], check=True, stdout=subprocess.DEVNULL)
        return dict(request=pin(path), profile=profile_pins[arm])

    for arm in profiles:
        write(destination / (arm + "-qualification.json"),
              dict(schema_version=1, timeout_seconds=2400, **invocation(arm, "qualification")))
    pairs = []
    for index, first in enumerate(("baseline", "candidate", "baseline")):
        pairs.append(dict(first=first, **{a: invocation(a, str(index)) for a in profiles}))
    bridge = Path(profiles["candidate"]["environment"]["FIREPROT_BRIDGE_LIBRARY"]).resolve()
    changes = {"backend": dict(baseline="native", candidate="hip_fused"),
               "invocation.profile_sha256": {a: p["sha256"] for a, p in profile_pins.items()},
               "launch.inputs.rust-bridge": dict(baseline=None, candidate=dict(
                   name="rust-bridge", canonical_path=str(bridge), sha256=pin(bridge)["sha256"]))}
    changes.update({"launch.environment." + flag: dict(baseline="0", candidate="1") for flag in FLAGS})
    specification = read(paired_root / "pairs.json")
    specification["execution"].update(pairs=pairs, timeout_seconds=10800, invocation_timeout_seconds=2400)
    specification["comparison"].update(expected_changes=changes, treatment_description=
        "Original native Torch under the common FireProt launcher versus qualified unroll4 plus HIP global bias packing/softmax. "
        "Strict FP32 N1536 R3/four trunk passes D200/S1 seed101; native data-module defaults preserved.")
    write(destination / "pairs.json", specification)
    write(destination / "preparation.json", dict(schema_version=1, cli=prep["cli"],
          paired_qualification=pin(paired_root / "collected.json"), source_preparation=pin(source_preparation),
          reference_npz=prep["reference_npz"], reference_cif=prep["reference_cif"],
          router=pin(router), reference_worker=pin(reference_worker), routes=routes,
          scope=specification["comparison"]["treatment_description"],
          status="Prepared only; both routed model gates are required before paired execution",
          files=[pin(p) for p in sorted(destination.rglob("*")) if p.is_file()]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("paired_root", "destination", "output_root", "reference_worker"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.paired_root, args.destination, args.output_root, args.reference_worker)
