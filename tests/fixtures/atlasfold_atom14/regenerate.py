"""Regenerate offline writer fixtures using the pinned AtlasFold environment.

Usage: <atlasfold-python> regenerate.py <atlasfold-repository> <destination>
This writes synthetic data only and does not load model weights or run inference.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gemmi
import numpy as np

root, destination = map(Path, sys.argv[1:])
commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert commit == "8ab3aca0e18c8b814d5ca6756b2617a07d72c68d"
assert gemmi.__version__ == "0.7.5"
sys.path.insert(0, str(root / "src"))
from atlasfold.common import protein, residue_constants  # noqa: E402

destination.mkdir(parents=True, exist_ok=True)
sequence = "ACDEFGHIKLMNPQRSTVWYX"
coordinates = np.arange(len(sequence) * 42, dtype=np.float32).reshape(len(sequence), 14, 3) / np.float32(17) - np.float32(25)
coordinates.reshape(-1)[:15] = [0., -0., .0005, -.0005, 1.2345, -1.2345, .0115, -.0115, 123.4565, -123.4565, 1024.0005, -1024.0005, 1000000.125, -1000000.125, 1.1875]
prediction = protein.Protein.create("atom14_fixture", sequence, coordinates, np.full(len(sequence), 80, dtype=np.float32))
(destination / "prediction.cif").write_text(prediction.to_mmcif())
np.savez(destination / "coordinates.npz", coordinates=coordinates)

# Include ordinary coordinate scales, decimal midpoints, adjacent FP32 values,
# random finite bit patterns, signed zeros, subnormals and maximal finite values.
rng = np.random.default_rng(101)
bits = rng.integers(0, 2**32, 4096, dtype=np.uint32)
values = bits.view(np.float32)
values = values[np.isfinite(values)]
midpoints = np.array([n / 1000 + .0005 for n in range(-1000, 1000)], dtype=np.float32)
values = np.concatenate([values, midpoints, np.nextafter(midpoints, np.float32(np.inf)), np.nextafter(midpoints, np.float32(-np.inf)), np.array([0., -0., 1000000.125, -1000000.125, np.finfo(np.float32).max, -np.finfo(np.float32).max, np.finfo(np.float32).smallest_subnormal], dtype=np.float32)])
# Use the actual upstream writer, not a reimplementation of its formatter.
padded = np.zeros((len(values), 14, 3), dtype=np.float32)
padded[:, 0, 0] = values
rounding = protein.Protein.create("rounding_fixture", "G" * len(values), padded, np.full(len(values), 80, dtype=np.float32))
block = gemmi.cif.read_string(rounding.to_mmcif()).sole_block()
x = list(block.find_values("_atom_site.Cartn_x"))[::4]
expected = np.array([float(v) for v in x], dtype=np.float32)
rounding_rows = [[int(a), int(b)] for a, b in zip(values.view(np.uint32), expected.view(np.uint32), strict=True)]
(destination / "rounding.json").write_text(json.dumps(rounding_rows, separators=(",", ":")) + "\n")
metadata = {
    "sequence": sequence,
    "upstream_commit": commit,
    "gemmi_version": gemmi.__version__,
    "atom_count": sum(len(residue_constants.residue_atoms[residue_constants.restype_1to3[a]]) for a in sequence),
    "rounding_cases": len(rounding_rows),
    "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ["src/atlasfold/common/protein.py", "src/atlasfold/common/file_io.py", "src/atlasfold/common/residue_constants.py", "src/atlasfold/runner.py"]},
    "fixture_sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in ["prediction.cif", "coordinates.npz", "rounding.json"]},
}
(destination / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata))
