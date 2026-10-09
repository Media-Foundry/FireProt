"""Experimental common entry for a direct native-versus-optimized comparison.

Preparation copies this file to fireprot_workers/openfold3.py, the unchanged
audited reference worker to openfold3_reference.py, and one qualified optimized
worker to openfold3_optimized.py. Both arms use the same tree. Only the selected
worker is imported; its source digest and module origin must match the manifest.
This helper is not connected to the maintained default or a GPU experiment yet.
"""
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path

REFERENCE_SHA256 = "c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e"
ROUTES = {"native": "openfold3_reference", "hip_fused": "openfold3_optimized"}


def predict(request, channel):
    backend = request["backend"]
    if request["workload"]["model"] != "openfold3" or backend not in ROUTES:
        raise ValueError("Unsupported OpenFold3 comparison route")
    root = Path(__file__).resolve().parent
    routes = json.loads((root / "openfold3_routes.json").read_text())
    if routes.get("schema_version") != 1 or set(routes.get("workers", {})) != set(ROUTES):
        raise ValueError("Malformed OpenFold3 comparison manifest")
    name = ROUTES[backend]
    declared = routes["workers"][backend]
    if declared["module"] != name:
        raise ValueError("Comparison worker name differs from the fixed route")
    source = root / (name + ".py")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != declared["sha256"] or (backend == "native" and digest != REFERENCE_SHA256):
        raise RuntimeError("Comparison worker source mismatch")
    selected = import_module("." + name, package=__package__)
    if Path(selected.__file__).resolve() != source:
        raise RuntimeError("Comparison worker import origin mismatch")
    bundle = selected.predict(request, channel)
    receipt = dict(schema_version=1, backend=backend,
                   selected_worker=dict(module=name, path=str(source), sha256=digest),
                   reference_worker_sha256=REFERENCE_SHA256,
                   scope="Only the selected worker is imported by this router; no request rewriting")
    Path(os.environ["FIREPROT_OUTPUT_DIR"], "openfold3-route.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    return bundle
