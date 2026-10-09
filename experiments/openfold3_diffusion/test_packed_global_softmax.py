"""Host guards for the separately qualified HIP bias packer."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_module():
    name = "experiments.openfold3_diffusion.packed_global_softmax"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("packed_global_softmax.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tampered_library_rejected_before_dlopen(tmp_path):
    module = load_module()
    binary = tmp_path / "invalid.so"
    binary.write_bytes(b"changed library")
    receipt = tmp_path / "build.json"
    receipt.write_text(json.dumps(dict(library=str(binary), library_sha256="0" * 64)))
    with pytest.raises(ValueError, match="digest mismatch"):
        module.PackedGlobalSoftmaxDispatcher(None, SimpleNamespace(device=0), receipt)


def test_cpu_operand_cannot_launch_packer():
    module = load_module()
    dispatch = object.__new__(module.PackedGlobalSoftmaxDispatcher)
    dispatch.kernel = SimpleNamespace(device=0)
    with torch.inference_mode(), pytest.raises(ValueError, match="Unsupported"):
        dispatch.pack_bias(torch.zeros(1))


def test_invalid_tile_rejected_before_loading(tmp_path):
    module = load_module()
    with pytest.raises(ValueError, match="64 or 128"):
        module.PackedGlobalSoftmaxDispatcher(None, SimpleNamespace(device=0), tmp_path / "missing.json", 32)
