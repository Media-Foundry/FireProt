"""Numerical comparison helpers for Protenix inference outputs."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import numpy as np


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten numeric JSON leaves into stable dotted/indexed paths."""
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_numeric(value[key], child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(flatten_numeric(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def aligned_coordinate_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float | int | bool]:
    """Compare coordinate arrays before and after rigid Kabsch alignment."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Coordinate shape mismatch: {reference.shape} != {candidate.shape}"
        )
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape [N, 3], got {reference.shape}")
    if reference.shape[0] == 0:
        raise ValueError("Coordinate arrays must not be empty")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return {
            "atom_count": reference.shape[0],
            "finite": False,
            "raw_rmsd_angstrom": math.inf,
            "aligned_rmsd_angstrom": math.inf,
            "max_abs_coordinate_delta_angstrom": math.inf,
        }

    delta = candidate - reference
    mobile = candidate - candidate.mean(axis=0)
    target = reference - reference.mean(axis=0)
    left, _, right_transpose = np.linalg.svd(mobile.T @ target)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    aligned_delta = mobile @ rotation - target
    return {
        "atom_count": reference.shape[0],
        "finite": True,
        "raw_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum(delta * delta, axis=-1)))
        ),
        "aligned_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum(aligned_delta * aligned_delta, axis=-1)))
        ),
        "max_abs_coordinate_delta_angstrom": float(np.max(np.abs(delta))),
    }


def confidence_metrics(
    reference: Any, candidate: Any
) -> dict[str, float | int | list[str] | bool]:
    """Compare every numeric confidence leaf and report schema differences."""
    reference_values = flatten_numeric(reference)
    candidate_values = flatten_numeric(candidate)
    reference_keys = set(reference_values)
    candidate_keys = set(candidate_values)
    common = sorted(reference_keys & candidate_keys)
    deltas = [
        abs(candidate_values[key] - reference_values[key]) for key in common
    ]
    finite = all(
        math.isfinite(reference_values[key]) and math.isfinite(candidate_values[key])
        for key in common
    )
    return {
        "numeric_leaf_count": len(common),
        "finite": finite,
        "max_abs_delta": max(deltas, default=0.0),
        "mean_abs_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "missing_from_candidate": sorted(reference_keys - candidate_keys),
        "extra_in_candidate": sorted(candidate_keys - reference_keys),
    }


def load_cif_coordinates(path: pathlib.Path) -> np.ndarray:
    """Load model-1 coordinates lazily so array-only tests need no Biotite."""
    try:
        from biotite.structure.io.pdbx import CIFFile, get_structure
    except ImportError as error:
        raise RuntimeError(
            "Biotite is required to compare Protenix CIF predictions"
        ) from error
    return get_structure(CIFFile.read(path), model=1).coord.astype(np.float64)


def load_ground_truth_coordinates(path: pathlib.Path) -> np.ndarray:
    """Load only experimentally mapped coordinates from a case mapping."""
    with np.load(path) as mapping:
        coordinates = np.asarray(mapping["coordinates"])
        mask = np.asarray(mapping["mask"], dtype=bool)
    return coordinates[mask]


def prediction_paths(
    root: pathlib.Path, case: str, seed: int, sample: int
) -> tuple[pathlib.Path, pathlib.Path]:
    prediction_dir = root / case / f"seed_{seed}" / "predictions"
    return (
        prediction_dir / f"{case}_sample_{sample}.cif",
        prediction_dir / f"{case}_summary_confidence_sample_{sample}.json",
    )


def compare_prediction(
    reference_root: pathlib.Path,
    candidate_root: pathlib.Path,
    *,
    case: str,
    seed: int,
    sample: int,
    coordinate_tolerance: float,
    confidence_tolerance: float,
) -> dict[str, Any]:
    reference_cif, reference_confidence = prediction_paths(
        reference_root, case, seed, sample
    )
    candidate_cif, candidate_confidence = prediction_paths(
        candidate_root, case, seed, sample
    )
    for path in (
        reference_cif,
        reference_confidence,
        candidate_cif,
        candidate_confidence,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    coordinates = aligned_coordinate_metrics(
        load_cif_coordinates(reference_cif), load_cif_coordinates(candidate_cif)
    )
    reference_json = json.loads(reference_confidence.read_text(encoding="utf-8"))
    candidate_json = json.loads(candidate_confidence.read_text(encoding="utf-8"))
    confidence = confidence_metrics(reference_json, candidate_json)
    passed = (
        bool(coordinates["finite"])
        and bool(confidence["finite"])
        and coordinates["aligned_rmsd_angstrom"] <= coordinate_tolerance
        and confidence["max_abs_delta"] <= confidence_tolerance
        and not confidence["missing_from_candidate"]
        and not confidence["extra_in_candidate"]
    )
    return {
        "case": case,
        "seed": seed,
        "sample": sample,
        "reference": {
            "cif": str(reference_cif),
            "confidence": str(reference_confidence),
        },
        "candidate": {
            "cif": str(candidate_cif),
            "confidence": str(candidate_confidence),
        },
        "coordinate_tolerance_angstrom": coordinate_tolerance,
        "confidence_tolerance": confidence_tolerance,
        "coordinates": coordinates,
        "confidence": confidence,
        "passed": passed,
    }
