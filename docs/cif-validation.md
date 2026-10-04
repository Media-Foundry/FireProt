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

Current AtlasFold inference requires output bundle schema 3; Protenix and
OpenFold3 require schema 2. Every prediction names a
`structure` path, relative to the run directory, in addition to its NPZ `archive`.
Rust requires the structure in the supervisor's hash-verified artifact list,
rejects reused/escaping paths and unassociated CIF artifacts, and parses the file
before accepting the run. Prediction atom IDs must be consecutive from one.

- Protenix: use the pinned upstream dumper's rank mapping to associate original
  sample indices with filenames. Parsed CIF coordinates must equal that sample's
  stored coordinates exactly.
- OpenFold3: bind the upstream seed/sample filename to its original sample index;
  parsed coordinates must equal the stored sample exactly.
- AtlasFold schema 3: retain the consumed input bytes as an `input_snapshot`
  artifact. Rust independently checks its SHA-256 against the request input pin,
  parses the sequence, and checks every sample's exact atom14-to-CIF correspondence.
  The original request path determines JSON versus FASTA parsing; its file is not
  reopened during archive replay. Raw output parity remains a separate comparison.
  Historical schema 2 only checks the structural contract and bounds the atom
  count by atom14 storage.

Historical archive replay continues to accept schema 1 and AtlasFold schema 2.
They do not acquire stronger guarantees retroactively. New inference cannot use
an old schema to bypass its model's current acceptance requirements. Schema 3 is
currently reserved for AtlasFold; other model families retain schema 2.

The input snapshot must be a contained, independently hash-verified artifact.
Input parsing is bounded to 16 MiB and requires a nonempty named monomer with the
worker's accepted residue alphabet. Every requested sample must match this same
sequence. A missing snapshot, wrong digest, multimer, invalid residue, escaping
path or mismatched CIF fails acceptance. A caller can relocate an intact accepted
archive without restoring the old input path.

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

## AtlasFold atom14 correspondence primitive

`fireprot_validation::atlasfold::validate_atom14` now checks the complete atom
identity/order and serialized coordinates against an explicitly supplied monomer
sequence and FP32 `[N,14,3]` tensor. It uses the pinned AtlasFold atom order,
including five atoms for `X`/`UNK`, chain `A`, one-based residue/atom numbers,
and no insertion codes. It rejects missing, additional or misidentified atoms.
Unused atom14 slots have no physical CIF counterpart; the separate full-array
comparison continues to cover them.

The writer has two formatting stages: Gemmi 0.7.5 serializes double coordinates
with `%.9g` through stb_sprintf, then AtlasFold parses those strings and formats
three decimal places using Python. Decimal midpoint behavior differs between
stb_sprintf and Rust's formatter. The Rust implementation reproduces the first
rounding from the exact FP32 decimal expansion, applies the second rounding,
and compares the shared CIF reader's FP32 values exactly. It introduces no new
coordinate tolerance. Source references:
[Gemmi coordinate writer](https://github.com/project-gemmi/gemmi/blob/v0.7.5/src/to_mmcif.cpp),
[Gemmi number formatting](https://github.com/project-gemmi/gemmi/blob/v0.7.5/include/gemmi/sprintf.hpp).

The offline fixture under `tests/fixtures/atlasfold_atom14` was generated by the
pinned upstream writer. It covers all 21 accepted residue symbols and 10,087
coordinate cases, including random FP32 bit patterns, decimal boundaries and
adjacent values, signed zeros, subnormals and large finite values. The generator,
upstream source hashes and fixture hashes are retained. CPU tests also reject
identity/sequence changes, changed written coordinates and missing/extra atoms.
All 31 validation-crate tests and four relevant Rust 1.85 tests pass.

The diagnostic example `validate_atlasfold_structure SEQUENCE ARRAY_NPZ
PREDICTION_CIF` passes saved N1000 and N1536 outputs, checking 7,978 and 12,138
atoms respectively. See `provenance/rust_atlasfold_atom14_mapping.json` for file
identities. These are offline mapping checks, not new model or timing results.

The primitive is now called by schema 3 bundle validation after authenticating and
parsing the retained input bytes. Current inference rejects AtlasFold schema 2;
archive replay retains the earlier checks. CPU regressions cover JSON/FASTA,
multiple samples, wrong input pins, missing/unverified/escaping snapshots, changed
sequences, a corrupted second sample, schema downgrade and archive relocation
with the original input absent.

### Schema 3 integration evidence

Four requests (six predictions) pass current acceptance and independent archive
replay on DiamondHill: N1000 native, N1536 fused HIP, and N1000 two-sample
chunked/fused HIP. All use the original dual checkpoints, native precision,
seed 102 and four additional recycles. N1000 uses 30 diffusion steps and N1536
uses 100. Both samples of the chunked/fused pair match exactly across coordinates,
pLDDT, PAE and pTM. Single-sample runs match their existing native references
exactly. The two-sample inputs use FASTA; the single-sample inputs use JSON.

A relocated real two-sample archive replays with its original input absent.
A separate tampered copy fails even after resealing the snapshot's output hash
and repinning the receipt: its bytes differ from the original request input pin.
The successful relocated archive is preserved separately from this negative case.

Two unsuccessful trials remain explicit in the evidence. A test-harness error
compared a 30-step N1536 request against the 100-step reference; that comparison
is excluded and the corrected 100-step request passes. Unchunked native N1000
with two samples exhausted GCD memory and produced no accepted receipt. The
existing chunked route supplies the bounded native-math two-sample reference;
length, sample count, steps and seed were retained. Neither failure was handled
by relaxing a numerical gate.

A 165-test Rust workspace run and 272-test Python run pass (110 Python skips).
A subsequent FASTA whitespace compatibility regression first failed in Rust and
passed in Python, then passed after matching Python's strip semantics. Final
validation-crate runs pass all 36 tests on stable and Rust 1.85; the additional
Python regression passes separately. Workspace Clippy and format checks pass.
The 101 checked Rust/worker source files match DiamondHill. Final-validator archive
replay covers all four runs after the whitespace-only parser correction. Runs
used HIP 6 / PCI 8e / SMI 4 and HIP 7 / PCI 93 / SMI 5 with continuous isolation.
Some runs overlap on separate GCDs; this establishes acceptance, not a performance
comparison. See `provenance/rust_atlasfold_input_binding.json` for exact requests,
receipts, source hashes, comparisons and retained failure records.
