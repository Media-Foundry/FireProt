"""Discriminating preparation guards; these do not qualify GPU execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.openfold3_diffusion import prepare_native_comparison as preparation


def test_native_flags_are_disabled_without_mutating_source(tmp_path):
    source = dict(workers="qualified", environment={key: "1" for key in preparation.FLAGS},
                  environment_inventory=dict(path="pinned", sha256="inventory"))
    source["environment"]["OPENFOLD_CACHE"] = "data-cache"
    original = deepcopy(source)
    profiles = preparation.comparison_profiles(source, tmp_path)
    assert source == original
    assert profiles["baseline"]["workers"] == profiles["candidate"]["workers"] == str(tmp_path)
    assert all(profiles["baseline"]["environment"][flag] == "0" for flag in preparation.FLAGS)
    assert all(profiles["candidate"]["environment"][flag] == "1" for flag in preparation.FLAGS)
    assert profiles["baseline"]["environment"]["OPENFOLD_CACHE"] == "data-cache"
    assert profiles["baseline"]["environment_inventory"] == source["environment_inventory"]


def test_router_preserves_both_worker_bodies(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    reference = repo / "python/fireprot_workers/openfold3.py"
    router = Path(__file__).with_name("native_router.py")
    optimized = b"# Qualified optimized body, including absolute artifact paths.\n"
    (tmp_path / "openfold3.py").write_bytes(optimized)
    routes = preparation.install_routes(tmp_path, reference, router)
    assert (tmp_path / "openfold3_optimized.py").read_bytes() == optimized
    assert (tmp_path / "openfold3_reference.py").read_bytes() == reference.read_bytes()
    assert (tmp_path / "openfold3.py").read_bytes() == router.read_bytes()
    assert routes["workers"]["hip_fused"]["sha256"] == hashlib.sha256(optimized).hexdigest()
    assert routes["workers"]["native"]["sha256"] == preparation.REFERENCE_SHA256
    with pytest.raises(ValueError, match="already installed"):
        preparation.install_routes(tmp_path, reference, router)


def test_modified_native_reference_cannot_replace_qualified_entry(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    original = package / "openfold3.py"
    original.write_text("optimized")
    reference = tmp_path / "reference.py"
    reference.write_text("modified native")
    with pytest.raises(ValueError, match="audited unchanged"):
        preparation.install_routes(package, reference, Path(__file__).with_name("native_router.py"))
    assert original.read_text() == "optimized"
    assert not (package / "openfold3_optimized.py").exists()


def test_regressed_pair_blocks_preparation_before_any_copy(tmp_path):
    source = tmp_path / "paired"
    source.mkdir()
    report = dict(correctness_passed=True, runs=[{}] * 6,
                  performance=dict(paired_speedups=[1.1, .999, 1.2]))
    (source / "collected.json").write_text(json.dumps(report))
    target = tmp_path / "prepared"
    with pytest.raises(ValueError, match="every E2E pair"):
        preparation.prepare(source, target, tmp_path / "outputs", tmp_path / "reference")
    assert not target.exists()
