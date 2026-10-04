# Rust numerical validation

`fireprot-validation` implements the numerical comparison primitives needed by
the Rust migration. An output-bundle gate now validates model-required fields, sample identity and
original dtype metadata through the Rust supervisor. Model-worker integration
and parity remain in progress.

## Array identity and files

Arrays retain dtype, full shape and canonical little-endian C-order value bytes.
Exact comparison requires all three to match, including signed zero. Floating
arrays are checked for NaN/Inf at construction. Explicit absolute-tolerance
comparison accepts only floating arrays and finite nonnegative tolerances.
Integers are compared exactly without conversion through f64.

NPY and NPZ readers preserve primitive float/integer/bool types and normalize
endianness and Fortran layout. They reject object/structured arrays, unsupported
dtypes, empty arrays, mismatched byte counts, duplicate archive names and
archives exceeding a caller-specified uncompressed-byte limit. This limit bounds
input size, not total peak memory during decoding/comparison.

In-memory tensors support native BF16 bytes. Standard NumPy's opaque BF16
storage is not inferred as BF16 by the file reader: workers must provide explicit
original dtype metadata. Historical OpenFold3 NPZ captures converted BF16 to
FP32 without that metadata; comparisons of those files establish stored-array
agreement only. They cannot reconstruct the missing original tensor dtype.

## Coordinates and confidence

Kabsch alignment uses f64 arithmetic and a proper rotation (determinant +1).
Reflection cannot be used to improve agreement. Empty/mismatched coordinates,
nonfinite values, overflow or a failed SVD are errors. The report includes raw
RMSD, aligned RMSD, maximum raw coordinate delta and atom count.

The Protenix predicate preserves the existing aligned-RMSD and maximum numeric
confidence-delta gates. It also requires matching numeric confidence fields.
Callers supply the predeclared tolerances; the established Protenix comparison
uses 1e-3 Angstrom aligned RMSD and 1e-3 confidence delta. No tolerance is derived
from candidate errors. Confidence paths encode object/array segments distinctly,
so object keys cannot collide with array indices or nested dotted names. Empty
numeric confidence objects fail rather than vacuously passing.

Experimental coordinates come exclusively from `coordinates[mask]` in a mapping
archive. `reference` is never selected as experimental ground truth. Masks must
be boolean and match the coordinate row count.

## Verification and remaining work

Tests include NumPy-generated big-endian/Fortran data, compressed archives,
masked experimental coordinates and an independently computed NumPy SVD result.
They also cover reflections, degenerate coordinates, signed zero, dtype/shape
mismatches, half precision, missing confidence fields and malformed inputs.
Fixture generation uses Python/NumPy only to regenerate test data; Rust tests and
runtime comparison use the committed fixtures directly.

```bash
cargo test -p fireprot-validation
cargo run --release --example compare_npz -- reference.npz candidate.npz
```

The example checks every archive key with exact comparison and returns nonzero
on failure. Broader model output schemas, CIF handling and repeated-timing statistics
remain migration work. Saved-output replay is not a new inference benchmark.

NPY parsing uses [npyz](https://docs.rs/npyz/0.9.1/npyz/), and coordinate
alignment uses [nalgebra decompositions](https://www.nalgebra.rs/docs/user_guide/decompositions_and_lapack/).
