"""Require route evidence to be sealed and to retain its accepted bytes."""
import hashlib
import json

import pytest

from experiments.openfold3_diffusion.qualify_native_comparison import verify_sealed_outputs


def test_route_receipt_must_be_present_in_acceptance_archive(tmp_path):
    (tmp_path / "openfold3-route.json").write_text('{"backend":"native"}')
    (tmp_path / "accepted.json").write_text(json.dumps(dict(metadata=[], outputs=[])))
    with pytest.raises(ValueError, match="absent from the sealed"):
        verify_sealed_outputs(tmp_path, ["openfold3-route.json"])


def test_changed_route_bytes_fail_after_initial_success(tmp_path):
    path = tmp_path / "openfold3-route.json"
    path.write_text('{"backend":"native"}')
    receipt = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path / "accepted.json").write_text(json.dumps(dict(metadata=[], outputs=[receipt])))
    verify_sealed_outputs(tmp_path, [path.name])
    path.write_text('{"backend":"hip_fused"}')
    with pytest.raises(ValueError, match="Changed prerequisite"):
        verify_sealed_outputs(tmp_path, [path.name])
