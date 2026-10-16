# Model-resident protein language model batches

This opt-in path loads one model per request and executes Rust-planned tensor
microbatches. It supports standalone AtlasLM-3B, ESMC-fold2-6B, ESM2-3B and
ESM3-open. It does not change folding or the heterogeneous `infer-batch` command,
which still launches independent requests.

The latest ESMC/ESM2/ESM3 operator and loading cycle is documented in
[ESM optimization](esm-optimization.md), including ESM3's opt-in sequence-only
geometry simplification. The dated results below preserve the initial batch cycle.

## Input and request

Use the existing `fireprot esm REQUEST PROFILE` or
`fireprot atlaslm REQUEST PROFILE` command with a `plm` workload. Add an explicit
`batch` object, and pin a JSON input manifest:

```json
{
  "model": "plm",
  "model_name": "atlaslm-3b",
  "precision": "native",
  "batch": {
    "items": 4,
    "max_batch_size": 4,
    "max_padded_tokens": 4008,
    "grouping": "exact_length"
  }
}
```

The input format is separate from the single-sequence FASTA/Protenix JSON format:

```json
{
  "schema_version": 1,
  "sequences": [
    {"id": "protein-a", "sequence": "ACDE"},
    {"id": "protein-b", "sequence": "FGHI"}
  ]
}
```

`items` must equal the manifest length (two for the small input above). There are
1–256 items, each with 1–1000 residues from the 20 standard amino acids plus X.
IDs are unique, 1–128 ASCII letters, digits, periods, underscores or hyphens.
`max_batch_size` is 1–16; `max_padded_tokens` is 3–16032 and includes BOS/EOS.
An individual sequence must fit the token budget. These are contract bounds,
not a guarantee that every allowed configuration fits GPU memory.

Rust sorts by length with original index as a stable tie breaker. `exact_length`
combines only equal lengths; `length_sorted` explicitly permits padding. Both
respect the two microbatch limits. `max_batch_size: 1` reuses a resident model
while retaining singleton tensor shapes. The original input order and IDs are
preserved in outputs regardless of execution order. There is no feature cache.

## Numerical and output contract

The user-approved strict reference is **the official model using the identical
microbatch grouping**, including padding and native precision. Differences from
singleton inference are reported separately. A batch speedup is not a claim of
bitwise equality to singleton inference, or unchanged downstream folding quality.
Native batch GEMM/attention selection can change rounding even without padding.

Schema 6 binds the complete retained input snapshot to its SHA-256, recomputes
the Rust grouping plan, and requires every input exactly once in original order.
`sample` means the original input index in this schema; `input_id` preserves its
manifest identity. Each archive retains BOS/EOS, native embedding/logit dtypes,
and exact token IDs, with batch padding removed. BF16 is stored as raw uint16
bits. Wrong IDs, missing/reordered items, changed groups, padded output shapes,
nonfinite tensors and unverified artifacts fail acceptance. A failed worker
cannot seal a partially successful batch.

The native backend uses native model operators. Experimental FireProt batches
are restricted to N1000 and actual group sizes 1, 2 or 4 for AtlasLM, ESMC,
ESM2 and the new `hip_esm` ESM3 route. Other shapes fail explicitly.
Operator qualification and complete-invocation performance are separate gates.

## Initial native batching screen, 2026-10-16

HIP6 on DiamondHill, gfx90a MI250, native precision, seed102 for AtlasLM and
seed101 otherwise. Distinct deterministic rotations of the existing development
N1000 input were used; these are not an independent biological test set. Two
warmups per arm, AB/BA/AB order, three observations per arm per pair, unflushed
caches, resident model. Timings below exclude loading, tokenization, export and
validation and are **not complete invocation E2E**.

| Model | Four N1000 singleton forwards → native B4 | Throughput gain | Mixed 600/800/900/1000 gain |
| --- | ---: | ---: | ---: |
| AtlasLM-3B | 418.418 → 364.648 ms | 1.147x | 0.983x |
| ESMC-fold2-6B | 787.961 → 649.678 ms | 1.213x | 0.940x |
| ESM2-3B | 971.992 → 896.610 ms | 1.084x | 0.831x |
| ESM3-open | 471.956 → 404.125 ms | 1.168x | 0.937x |

Padding erased the throughput gain in all four mixed-length cases. Prefer exact
length groups until a workload-specific padding policy is measured. The warmed
throughput results do not establish the complete batch-job improvement from
avoiding repeated weight loading and asset verification.

For equal-length B4, maximum absolute singleton-to-batch differences were:

| Model | Embeddings | Sequence logits |
| --- | ---: | ---: |
| AtlasLM-3B | 1.262101 | 2.625 |
| ESMC-fold2-6B | 1.5 | No head |
| ESM2-3B | 0.000016213 | 0.000062943 |
| ESM3-open | 192 | 0.5 |

These raw quantities have different scales and meanings across models. ESM3
exports prenormalization embeddings. They are not coordinate errors or quality
scores. All reported outputs were finite and token identities exact; no
singleton-drift tolerance is inferred from these observations.

The Rust-supervised native entry subsequently passed all 12 checks: equal-length
B4, mixed-length B4, and resident B1 for each of the four models. All 48 item
archives match their corresponding retained native outputs bitwise. The first
deployment attempt failed before model execution because its source identity
directories were incomplete; its record is retained separately. The successful
snapshot is `outputs/experiments/plm-batch-20261016/qualified-v2` on DiamondHill.

Raw timings, per-item drift and entry receipts are retained in
[native batch evidence](../provenance/plm_batch_native_20261016.json).

## Opt-in batch operators

The new HIP C ABI uses an explicit batch dimension and resets the RoPE position
at each sequence boundary. AtlasLM SwiGLU flattens the batch/token rows while
preserving intermediate BF16 SiLU rounding. Existing single-sequence kernel
artifacts remain unchanged. ESM2 continues to omit only discarded attention-head
means and uses strict checkpoint-covered initialization. Every scope checks
operator coverage and restores the native methods on exit.

All B1/B2/B4 tests with real weights retain every native output bitwise. Packaged
B4 warm forward, including context entry/exit, improves 362.487 → 335.993 ms
(1.079x) for AtlasLM, 635.234 → 578.160 ms (1.099x) for ESMC, and
895.138 → 874.980 ms (1.023x) for ESM2. These are separate matched operator trials;
do not multiply their ratios by the earlier batching screen.
See [operator evidence](../provenance/plm_batch_operators_20261016.json).

### Same-grouping B4 complete-invocation results

Three fresh-process AB/BA/AB pairs on HIP6, four distinct N1000 sequences per
request, native precision, seed102 AtlasLM/seed101 otherwise, unflushed caches.
The reference is native **B4**, not four separate singleton processes. Both arms
use the same input, grouping, profile and assets; ESMC parallel full-file hashing
is shared. Every array in all 72 item archives is bitwise equal to the same
native B4 reference. Medians include preflight, loading, first forward, all four
output exports, Rust acceptance and shutdown.

| Model | Native B4 → FireProt B4 | Complete invocation speedup | Interpretation |
| --- | ---: | ---: | --- |
| AtlasLM-3B | 16.045 → 15.053 s | 1.066x | Eager GPU weight assignment plus exact batch HIP RoPE/SwiGLU |
| ESMC-fold2-6B | 24.138 → 25.041 s | 0.964x | No E2E improvement; retain native for a single B4 request |
| ESM2-3B | 40.347 → 20.026 s | 2.015x | Primarily checkpoint-covered initialization |
| ESM3-open | Not paired | — | Native batch entry qualified; no operator speedup claim |

ESMC paired ratios were 1.0072x, 0.9468x and 1.0078x; retain the full result,
including the slower median, rather than selecting the favorable pairs. ESM2
first-forward medians were 2.156 → 2.192 s; its cold E2E gain does not imply a
first-forward gain. Packaged warm forward is a separate measurement above.
See [rehashed paired evidence](../provenance/plm_batch_e2e_20261016.json).

### AtlasLM: amortizing repeated model loading

A separate three-pair AB/BA/AB comparison processes the same four distinct N1000
sequences using the **existing optimized singleton route** versus the optimized
resident B4 route. Both retain eager GPU assignment, native precision, exact HIP
operators, seed102, HIP6 and the same frozen worker/CLI/profile. Each singleton
matches its official singleton reference; the B4 output matches the official B4
reference. The reported singleton-to-batch drift is permitted by the explicit
grouping policy, not hidden behind an exact-output claim across grouping modes.

| Pair | Sum of four complete optimized singleton invocations | One complete optimized B4 invocation | Ratio |
| --- | ---: | ---: | ---: |
| 1, serial then batch | 57.305 s | 15.286 s | 3.749x |
| 2, batch then serial | 59.990 s | 15.204 s | 3.946x |
| 3, serial then batch | 58.967 s | 15.042 s | 3.920x |
| Median | **58.967 s** | **15.204 s** | **3.879x** |

This is an aggregate four-sequence throughput improvement from avoiding repeated
startup, integrity verification and model loading, together with tensor batching.
It is not a single-sequence latency or folding speedup. Each complete invocation
is individually timed/sealed by Rust; caller orchestration and post-measurement
verification are outside both arms. Filesystem/GPU caches were unflushed; no
output feature cache or observations were discarded. No corresponding aggregate
cold-process ratio is claimed for the other three models.
See [all 15 invocation records and acceptance pins](../provenance/atlaslm_batch_reuse_e2e_20261016.json).

## Run the frozen experimental entry

Use the matching Rust CLI and checked-in profile. Copy the request and give it a
fresh run ID/output directory before execution. The four-sequence development
input is `configs/inputs/plm-batch-n1000.json`; every item is N1000, so the example
`length_sorted` plan adds no padding.

```bash
# DiamondHill, after source/config synchronization; requests select HIP6.
CLI=/media/IntelSSD/fireprot/outputs/experiments/plm-batch-20261016/qualified-v3/target/release/fireprot
"$CLI" atlaslm configs/requests/plm-atlaslm-3b-b4-n1000-fireprot.json \
  configs/diamondhill/plm-atlaslm-3b-batch.json
"$CLI" esm configs/requests/plm-esm2-3b-b4-n1000-fireprot.json \
  configs/diamondhill/plm-esm2-3b-batch.json
"$CLI" esm configs/requests/plm-esmc-fold2-6b-b4-n1000-native.json \
  configs/diamondhill/plm-esmc-fold2-6b-batch.json
"$CLI" esm configs/requests/plm-esm3-open-b4-n1000-native.json \
  configs/diamondhill/plm-esm3-open-batch.json
```

The ESMC FireProt B4 example remains opt-in for warm/resident evaluation; the
single-B4 complete-invocation trial did not improve E2E. ESM3 uses the official
native factory and has no operator-acceleration claim. Default single-sequence
profiles and all folding settings remain separate.
