"""Reference isolation and source-pin regressions using actual Python imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def route(tmp_path, monkeypatch):
    name = "fireprot_route_test_package"
    package = ModuleType(name)
    package.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, name, package)
    wrapper = tmp_path / "openfold3.py"
    wrapper.write_bytes(Path(__file__).with_name("native_router.py").read_bytes())
    spec = importlib.util.spec_from_file_location(name + ".openfold3", wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("FIREPROT_OUTPUT_DIR", str(output))
    try:
        yield module, tmp_path, output
    finally:
        for key in list(sys.modules):
            if key.startswith(name + "."):
                del sys.modules[key]


def prepare(module, root, selected):
    workers = {}
    for backend, name in module.ROUTES.items():
        text = ("def predict(request, channel):\n    return (request, channel)\n"
                if backend == selected else "raise RuntimeError('unselected worker was imported')\n")
        path = root / (name + ".py")
        path.write_text(text)
        workers[backend] = dict(module=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root / "openfold3_routes.json").write_text(json.dumps(dict(schema_version=1, workers=workers)))
    return workers


@pytest.mark.parametrize("backend", ["native", "hip_fused"])
def test_only_selected_worker_imported_with_original_arguments(route, monkeypatch, backend):
    module, root, output = route
    workers = prepare(module, root, backend)
    monkeypatch.setattr(module, "REFERENCE_SHA256", workers["native"]["sha256"])
    request = dict(backend=backend, workload=dict(model="openfold3"))
    channel = object()
    result = module.predict(request, channel)
    assert result[0] is request and result[1] is channel
    receipt = json.loads((output / "openfold3-route.json").read_text())
    assert receipt["selected_worker"]["sha256"] == workers[backend]["sha256"]
    assert receipt["backend"] == backend


def test_reference_cannot_be_redeclared_as_optimized_source(route):
    module, root, output = route
    prepare(module, root, "native")
    with pytest.raises(RuntimeError, match="source mismatch"):
        module.predict(dict(backend="native", workload=dict(model="openfold3")), None)
    assert not (output / "openfold3-route.json").exists()


def test_modified_optimized_source_rejected_before_import(route):
    module, root, output = route
    prepare(module, root, "hip_fused")
    (root / "openfold3_optimized.py").write_text("raise RuntimeError('changed source imported')\n")
    with pytest.raises(RuntimeError, match="source mismatch"):
        module.predict(dict(backend="hip_fused", workload=dict(model="openfold3")), None)
    assert not (output / "openfold3-route.json").exists()


def test_reference_pin_matches_audited_repository_worker(route):
    module, _, _ = route
    root = Path(__file__).resolve().parents[2]
    reference = root / "python/fireprot_workers/openfold3.py"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == module.REFERENCE_SHA256
