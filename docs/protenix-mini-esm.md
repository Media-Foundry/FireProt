# Protenix Mini-ESM under Rust supervision

The worker integrates `protenix_mini_esm_v0.5.0` in FP32, preserving the upstream
ESM2-3B featurizer, total-cycle semantics and diffusion settings. ESM2-3B is a
separate dependency from AtlasFold's AtlasLM-3B. Neither is replaced or omitted.

## Explicit assets and invocation

Mini-ESM requests require three independently hashed checkpoint roles:

| Role | File under `/media/IntelSSD/fireprot/runtime/checkpoint/` | SHA-256 |
| --- | --- | --- |
| `model` | `protenix_mini_esm_v0.5.0.pt` | `1301bba9ad322518eace60fd244ded7904439f55317e90d402cc7a0c06026664` |
| `language_model` | `esm2_t36_3B_UR50D.pt` | `7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500` |
| `language_model_regression` | `esm2_t36_3B_UR50D-contact-regression.pt` | `4da500eab246481dc9c8c95bc7b1d02f2803d761c380b0e95186d4a07d0fc84e` |

The pinned upstream loader derives filenames from the folding checkpoint's
parent. The worker therefore requires both language-model artifacts at their
exact expected sibling paths; declaring an unrelated file cannot authorize an
implicit read. Rust requires all three roles and verifies all file hashes.
The same four CCD/data resources as Mini-default remain mandatory.

```bash
target/release/fireprot infer \
  configs/requests/protenix-mini-esm-n1000-native.json \
  configs/diamondhill/protenix-native.json

target/release/fireprot infer \
  configs/requests/protenix-mini-esm-n1000-fireprot.json \
  configs/diamondhill/protenix-all-published.json
```

Use a fresh output directory and run ID for each request. The example workload
is N1000, four total cycles, one diffusion step, one sample, FP32 and seed 101.
Requests longer than 4094 residues are rejected because the pinned upstream
featurizer would truncate them. This upper bound is not a tested capacity claim.

## Fresh features and validation

The Rust CLI launches every model worker with its new output directory as the
working directory. Upstream relative scratch files therefore remain inside the
run. Mini-ESM requires an absent `esm_embeddings` directory before inference and
uses the original upstream precompute path. There is currently no cross-request
ESM feature cache in the maintained Rust interface.

Before the folding model runs, the worker verifies the effective ESM model,
embedding width, complete sequence length, FP32 dtype, finite values and exact
equality between the generated embedding and the actual model input. Upstream
featurizer error files veto acceptance. This prevents swallowed feature-load
errors from becoming accepted zero-feature inference.

Each prediction NPZ retains `input.esm_token_embedding` with its original shape
and dtype. Rust independently requires FP32 `[N_token, 2560]`. Existing coordinates,
full confidence arrays, CIF associations and Protenix numerical gates remain.
The language model executes during data preparation, so `model_seconds` covers
the folding forward call; `request_seconds` also includes ESM loading/featurization.
Neither field is complete invocation E2E, which also includes Rust preflight.

## Maintained entry point

`scripts/infer.sh REQUEST PROFILE` now delegates to Rust. The original Python
launcher is archived under `.agents/legacy_protenix_infer.sh`; see
[entry-point migration](entrypoints.md). Complete framework acceptance still
requires the remaining environment and performance gates.

## N1000 numerical checks

Rust-supervised native and FireProt runs were compared with a separate upstream
`batch_inference.inference_jsons` invocation, using the same sequence, weights,
FP32, C4/S1/sample1, seed 101, hipBLAS and a freshly generated ESM embedding.
The independent reference and both worker runs produced exactly equal ESM input
features. Against that upstream reference:

| Route | Aligned RMSD (Angstrom) | Maximum summary confidence delta | Existing gates |
| --- | ---: | ---: | --- |
| Rust/native | 0.000170983 | 0.0000114441 | Pass |
| Rust/FireProt | 0.000123555 | 0.0000419617 | Pass |

The Rust-native versus Rust-FireProt comparison also passes (aligned RMSD
`0.000191780`, confidence delta `0.0000305176`). These are native-agreement
checks, not experimental structural-accuracy measurements. Full confidence
arrays are retained and checked for shape/finite values; they are not claimed
bitwise identical. Concurrent requests on distinct GCDs do not establish a
controlled speedup, and this integration does not change promoted GPU arithmetic.

The working-directory change was also checked with Mini-default FireProt N1000,
AtlasFold fused-HIP N1000 and strict deterministic OpenFold3 N1000. Mini-default
passes its existing coordinate/summary gates; AtlasFold and OpenFold3 retain
exact native array parity. All five Rust-supervised runs passed continuous
physical-device isolation and accepted-archive replay. CPU checks passed 125
Rust tests and 269 Python tests (110 environment/hardware-dependent skips), plus
nine request-contract tests on Rust 1.85.0; clippy and formatting checks passed.
See [`rust_mini_esm_integration.json`](../provenance/rust_mini_esm_integration.json).
