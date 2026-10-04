# Prediction CIF validation

Rust owns prediction CIF acceptance and the maintained Protenix structure
comparison command. Model workers retain each upstream writer and explicitly
associate its native CIF with the original sample index.

```bash
fireprot inspect-cif prediction.cif
fireprot compare-protenix-cif reference.cif reference-confidence.json \
  candidate.cif candidate-confidence.json
```

The comparison checks atom identities, then applies the existing `1e-3` Angstrom
aligned RMSD and `1e-3` maximum numeric confidence-delta gates. A failed gate or
invalid input returns a nonzero exit code. Coordinate parsing reproduces the
historical Biotite FP32 values promoted to f64 for alignment.

## Current inference and archived bundles

Current inference requires output bundle schema 2. Every prediction names a
`structure` path, relative to the run directory, in addition to its NPZ `archive`.
Rust requires the structure in the supervisor's hash-verified artifact list,
rejects reused/escaping paths and unassociated CIF artifacts, and parses the file
before accepting the run. Prediction atom IDs must be consecutive from one.

- Protenix: use the pinned upstream dumper's rank mapping to associate original
  sample indices with filenames. Parsed CIF coordinates must equal that sample's
  stored coordinates exactly.
- OpenFold3: bind the upstream seed/sample filename to its original sample index;
  parsed coordinates must equal the stored sample exactly.
- AtlasFold: the upstream writer omits unused atom14 slots and rounds coordinates
  to three decimal places. Current validation checks the structural contract and
  bounds the atom count by atom14 storage. It does **not** establish a full
  CIF-to-atom14 coordinate/sequence mapping. Raw output parity remains a separate
  array comparison.

Historical archive replay continues to accept schema 1. Such archives do not gain
schema 2 structure-validation guarantees retroactively. New inference cannot
use schema 1 to bypass CIF acceptance.

## Parser boundary

The reader uses pinned [pdbtbx 0.12.0](https://github.com/douweschulte/pdbtbx)
with strict mmCIF parsing. The supported prediction contract is one nonempty model,
finite coordinates, unique numeric atom IDs and no alternate conformers. Atom
identity includes chain, residue/insertion code, residue name, atom name and
chemical element. Atoms are ordered by numeric ID. Files larger than 64 MiB are
rejected; this bounds input size, not total parser allocation.

The dependency can panic on malformed mandatory atom fields. Rust catches unwinds
around parser-owned state and returns a schema error, discarding partial results.
It does not alter the global panic hook; a parser diagnostic can appear on stderr.
This recovery requires the standard unwind panic strategy used by this workspace;
it is not a sandbox against allocator exhaustion or aborting custom builds.

The `compression` dependency feature is enabled because the pinned release does
not compile with all default features disabled alone. Parallel parsing features
are not enabled. CIF syntax follows the
[IUCr CIF specification](https://www.iucr.org/what-we-do/digital-standards/cif/cif1/file-syntax).

## Verification scope

CPU regressions cover quoted values, multiline metadata, malformed mandatory
fields, nonfinite coordinates, multiple models, alternates, duplicate IDs, atom
identity, confidence gates and schema 1/2 acceptance boundaries. The synthetic
CLI worker emits a schema 2 bundle and a verified CIF.

Saved N1000 native CIFs from AtlasFold, Protenix and OpenFold3 contain 7,978, 7,979
and 7,978 atoms respectively. Rust coordinates match the historical Biotite reader
exactly for all three files. This saved-output check is not a new model inference
or a performance measurement.

Four full schema 2 requests completed on DiamondHill: AtlasFold N1000 native,
Protenix N1000 FireProt with one and two samples, and strict deterministic
OpenFold3 N1000. All passed structure acceptance and continuous physical GPU
isolation; all four accepted archives were independently replayed through
`collect-timings`. AtlasFold and OpenFold3 arrays match their native references
exactly. Protenix single-sample aligned RMSD is `8.9780e-5` Angstrom and maximum
summary confidence delta is `3.8147e-6`, passing unchanged gates. The Rust CIF
comparison agrees with the previous Python comparator to numerical rounding.
The two-sample run checks associations and does not claim separate native parity.

Local verification: 122 Rust tests, 29 Python worker tests, clean clippy/fmt,
and nine structure/bundle tests on Rust 1.85.0. See
[`rust_cif_integration.json`](../provenance/rust_cif_integration.json) for requests,
source hashes, physical devices, acceptance receipts and numerical results.
