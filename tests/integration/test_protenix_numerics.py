from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from fireprot.validation.protenix import (
    aligned_coordinate_metrics,
    compare_prediction,
    confidence_metrics,
    load_ground_truth_coordinates,
)


def test_alignment_removes_rigid_transform() -> None:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    candidate = reference @ rotation + np.array([4.0, -2.0, 7.0])

    metrics = aligned_coordinate_metrics(reference, candidate)

    assert metrics["raw_rmsd_angstrom"] > 1
    assert metrics["aligned_rmsd_angstrom"] < 1e-12
    assert metrics["finite"] is True


def test_alignment_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        aligned_coordinate_metrics(np.zeros((2, 3)), np.zeros((3, 3)))


def test_confidence_comparison_checks_all_numeric_leaves() -> None:
    reference = {"a": 1.0, "nested": [{"b": 2.0}], "ignored": "text"}
    candidate = {"a": 1.25, "nested": [{"b": 2.5}], "ignored": "changed"}

    metrics = confidence_metrics(reference, candidate)

    assert metrics["numeric_leaf_count"] == 2
    assert metrics["max_abs_delta"] == 0.5
    assert metrics["mean_abs_delta"] == 0.375
    assert metrics["missing_from_candidate"] == []
    assert metrics["extra_in_candidate"] == []


def test_ground_truth_loader_uses_coordinates_mask_not_reference(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "mapping.npz"
    coordinates = np.arange(15, dtype=np.float32).reshape(5, 3)
    reference = np.full((5, 3), -999.0, dtype=np.float32)
    mask = np.array([True, False, True, False, True])
    np.savez(path, coordinates=coordinates, reference=reference, mask=mask)

    loaded = load_ground_truth_coordinates(path)

    np.testing.assert_array_equal(loaded, coordinates[mask])


@pytest.mark.skipif(
    not (
        os.environ.get("FIREPROT_NATIVE_OUTPUT")
        and os.environ.get("FIREPROT_CANDIDATE_OUTPUT")
    ),
    reason="Set native and candidate output roots for the GPU integration gate",
)
def test_real_outputs_match_locked_tolerances() -> None:
    reference_root = pathlib.Path(os.environ["FIREPROT_NATIVE_OUTPUT"])
    candidate_root = pathlib.Path(os.environ["FIREPROT_CANDIDATE_OUTPUT"])
    cases = json.loads(os.environ.get("FIREPROT_NUMERICAL_CASES", '["1izr_L124"]'))
    results = [
        compare_prediction(
            reference_root,
            candidate_root,
            case=case,
            seed=101,
            sample=0,
            coordinate_tolerance=1e-3,
            confidence_tolerance=1e-3,
        )
        for case in cases
    ]
    assert all(result["passed"] for result in results), results
