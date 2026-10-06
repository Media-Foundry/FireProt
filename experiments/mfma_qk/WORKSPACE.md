# Private score workspace experiment

This experiment retains one initialized FP32 score tensor per `WorkspaceFusion`
instance. It uses the accepted prefetch-v11 kernel, original Q/K packing and
native PV. Strict deterministic execution and uninitialized-memory filling stay
enabled. The first `torch.empty` receives its normal NaN fill; subsequent calls
reuse allocated storage and fully overwrite its scores. This is an explicit
change in workspace/cache lifetime, not an equivalent allocation policy.

## Ownership and bounds

- Only independently screened B32/B256, N1536, four heads and K32 are accepted.
- The caller supplies a byte budget, capped at9GiB; the shape stays fixed.
- A workspace binds to its first device and stream and its constructor thread.
- Concurrent use, other threads/streams, graph capture, training and relaxed
  determinism/filling settings are rejected.
- Q/K packing, optional bias-contiguous conversion and native PV are inside the
  call. Only PV output escapes; a storage check rejects alias with the scores.
- Same-stream launch ordering places PV reads before any subsequent overwrite.
- Explicit close synchronizes the owning device and releases the retained tensor.
- A submission-path error prevents further calls through that instance.

These are standalone tensor-boundary experiments. There is no shared framework
workspace manager or model adapter integration. Rust should own any future
resource policy; model lifecycle and tuner compatibility must be established
before selecting this in inference.

## Validation and measurement

`benchmark_workspace.py CONTROLS NEW_DESTINATION --batch 32` first tests three
seeds101/102/103 and both bias layouts, including a fully masked outer item.
Before reuse it poisons the old scores with NaNs or a finite sentinel. Every
probability bit and PV output bit is compared with native Torch. Retained prior
public results must remain unchanged. GPU guards reject a second thread and
stream without submissions/reallocations. First-allocation call time and Torch
allocation peak are recorded separately from the warm measurements.

Five arms are interleaved in balanced cyclic/reversed order: native, split HIP,
fresh fusion, retained workspace, retained workspace with bias normalization.
All packing and any bias conversion remain inside timing. Every warm and timed
output is checked bitwise outside the interval. There are100 timed calls per
batch. Each workspace must report one allocation and30 calls, then release its
entire retained tensor. Two workspaces persist during all arms, adding2.25GiB
at B32 or18GiB at B256. This common retained memory is included in reported live
and peak Torch allocations; it does not establish a model memory benefit.
No cache flush or continuous device isolation monitor is provided by this
standalone script. A controller records idle-device preflight.

CPU tests cover byte-budget rejection before library loading, thread ownership,
concurrency/closed-instance rejection, strict-setting enforcement and rejection
of non-HIP operands before dispatch. GPU correctness and timing results belong
in separate pinned provenance; CPU tests do not establish GPU acceptance.

## Completed B32/B256 result

Both batches pass twelve full-score/PV qualifications, ten prior-result
immutability checks, thread/stream rejection gates,100 timed-output checks,
and exact allocation/release accounting. Each instance allocates once, serves
30 calls and releases its entire retained tensor. Six CPU rejection tests pass.

| Batch / bias | Split HIP (ms) | Fresh fusion (ms) | Workspace (ms) | Workspace + contiguous bias (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32 / contiguous | 5.528 | 6.196 | 5.359 | 5.360 |
| 32 / transposed | 5.827 | 8.328 | 7.482 | 5.663 |
| 256 / contiguous | 43.469 | 48.942 | 41.914 | 42.093 |
| 256 / transposed | 43.774 | 65.783 | 58.702 | 42.384 |

The normalized workspace reduces warm B256 time3.17% contiguous and3.18%
transposed versus split HIP. First workspace calls with contiguous bias cost
49.13/49.19ms at B256 (6.55/6.31ms at B32), including initial allocation/fill;
these are single observations with other Torch operations already warm.
The retained byte cost remains9GiB per B256 instance. Model peak memory,
chunk tuning, lifecycle and E2E benefit are untested, so no model path is changed.

Evidence: `provenance/openfold3_qk_workspace_benchmark.json`. Remote archive
`qk-workspace-v1-results.tar.gz`, SHA256
`1ee16f1fb30c8afb315d2920b988dc1ea3b9e32b6299b379c7f4b22034100d54`.

## Combined kernel comparison

Controls may provide `workspace_reference` with an independently pinned kernel
and its successful numerical screen. In that mode both retained arms normalize
bias inside the call, and the report captures both kernel receipts. This allows
same-run comparison of prefetch-v11 against combined-v12 under the same workspace
policy. Native, split HIP and fresh combined fusion remain the other three arms.

Both batches again pass all full-score/output/previous-result and ownership
checks,100 timed calls each, and one-allocation/30-call/release accounting.

| Batch / bias | Split HIP (ms) | Prefetch workspace (ms) | Combined workspace (ms) |
| --- | ---: | ---: | ---: |
| 32 / contiguous | 5.529 | 5.358 | 4.918 |
| 32 / transposed | 5.830 | 5.662 | 5.229 |
| 256 / contiguous | 43.481 | 41.863 | 38.500 |
| 256 / transposed | 43.793 | 42.173 | 38.792 |

B256 time falls8.03%/8.02% versus prefetch workspace and11.46%/11.42% versus
split HIP (contiguous/transposed). The9GiB retained-per-instance cost remains.
No model/E2E acceptance is implied. Evidence:
`provenance/openfold3_qk_workspace_combined_benchmark.json`; archive
`qk-workspace-combined-v12-results.tar.gz`, SHA256
`fcc56b9e5efd0b3683766fb46a129035e597709b5c82c61cc8d336dd23a67a5b`.

Before model integration, compare larger outer batches against full-batch
reference QK/PV while reusing a bounded B256 workspace. Changing the outer
batch can change BLAS selection, so output equivalence must be measured. Keep
normalization, slicing/assembly and workspace initialization inside their stated
timing scopes. Adapter guards must retain existing rejection of lazy negative/
conjugate views and unmatched layouts. Model memory, chunk tuner behavior,
workspace teardown and full11-array/CIF checks are required separately.

## B512/B1024 microbatch qualification

`MicrobatchFusion` is a separate experimental wrapper accepting only B512 and
B1024. It owns one B256 workspace, normalizes bias once per complete invocation,
executes complete B256 QK/softmax/PV slices and concatenates their PV results.
Strict settings remain enabled. Plain FP32 tensors, supported broadcasts/layouts,
thread/device/stream ownership and rejection of lazy negative/conjugate views
are checked. B1536 and unmatched batches remain unsupported.

The reference retains full-batch native QK and PV and uses the already accepted
in-place HIP softmax. It is not a fresh out-of-place Torch-softmax comparison;
that reference would exceed the64GiB device at B1024. The captured full-shape
QK/PV kernel symbols match the prior B256 split trace. Numerical checks remain
the acceptance gate, independent of kernel names.

Each batch passes six different-seed/layout qualifications, broadcast and
per-outer masks, a fully masked item, all probability bits, all output bits,
output strides and unchanged prior results. Old workspace scores are poisoned
before each diagnostic microbatch. Forty alternating timed calls per batch
(ten per arm/layout) also pass. The wrapper makes31 complete calls and62/124
kernel calls, allocates once and releases exactly9GiB. Thirteen CPU rejection
tests pass across workspace and microbatch modules.

| Batch / bias | Full-batch split HIP (ms) | Microbatch (ms) |
| --- | ---: | ---: |
| 512 / contiguous | 87.054 | 77.389 |
| 512 / transposed | 87.307 | 77.650 |
| 1024 / contiguous | 173.873 | 154.510 |
| 1024 / transposed | 174.167 | 154.751 |

This is about11.1% less warm operator time, including bias conversion and result
assembly. First workspace calls measure84.58ms/161.59ms for B512/B1024, with
other operations already warm. Both timed arms retain the same9GiB workspace.
B1024 median Torch peak allocation is49.57GiB for the full-reference arm and
13.57–13.61GiB for microbatching under these conditions; these are not model
memory or total device-memory measurements. There is still no model/E2E result.

The first attempt stopped before candidate execution because its diagnostic
script omitted the CPU annotation required by the trace extractor. It is retained
as `qk-microbatch-v1-failed-trace.tar.gz`; the corrected attempt uses a new root.
Evidence: `provenance/openfold3_qk_microbatch_benchmark.json`, with both archives,
raw traces, full measurements and source pins. Successful archive
`qk-microbatch-v2-results.tar.gz`, SHA256
`916accf6a4e6eaa34f4258671a267092753bfe1d188da07df2003343e31cd502`.

## Recorded model bias layout

The historical model trace in `provenance/openfold3_attention_trace_n1536.json`
records bias strides `[4,1,6144,4]`. New head-last tests reproduce those strides
and the query/key transpose `[4,1,4,6144]`. `bias_layout_supported` accepts these
dense layouts only when the caller normalizes bias before the packed ABI.
Overlapping/gapped layouts and unresolved negative/conjugate views are rejected.
The direct workspace also now rejects tensor subclasses and unresolved views.

Use `--bias-layouts head-last` in either benchmark. Direct comparisons require
both retained arms to normalize bias; fresh fusion explicitly normalizes inside
its timed call too. Raw kernel arithmetic and its packed ABI are unchanged.
All four batch sizes pass full probability/output checks, previous-result and
release accounting; B32/B256 additionally repeat thread/stream rejection tests.
All280 timed calls pass bitwise. Nineteen CPU rejection/layout tests pass.

| Batch / bias | Split HIP (ms) | Combined workspace/microbatch (ms) |
| --- | ---: | ---: |
| 32 / head-last | 5.718 | 5.120 |
| 32 / head-last transposed | 6.095 | 5.503 |
| 256 / head-last | 43.663 | 38.691 |
| 256 / head-last transposed | 43.998 | 39.016 |
| 512 / head-last | 87.219 | 77.548 |
| 512 / head-last transposed | 87.546 | 77.910 |
| 1024 / head-last | 174.044 | 154.595 |
| 1024 / head-last transposed | 174.416 | 155.119 |

Bias conversion and output assembly costs remain included. These synthetic
layout checks do not establish model memory or E2E benefit. Evidence:
`provenance/openfold3_qk_headlast_benchmark.json`; successful archive
`qk-headlast-v1-results.tar.gz`, SHA256
`2722f1a18d94292209c52f91a5bf5565b2fe64b0558c2a554798fa36b3c9210b`.

## Rust policy and shared model owner (2026-10-06)

The opt-in model owner is implemented in `model_workspace.py`, but is not yet
installed into model dispatch or GPU-qualified as a mixed-batch lifetime.
`fireprot-core::workspace::WorkspaceLease` owns checked size/budget accounting,
thread/device/stream binding and allocation/use/drain/release transitions. The
additive CPython bridge exposes acknowledgments; it neither holds a tensor nor
independently verifies GPU completion.

`ModelWorkspaceFusion` owns one B256 pool for outer B256/B512/B1024. It allocates
through ordinary `torch.empty` under strict filling, normalizes bias once per
complete call, and queues each kernel/native-PV pair on the bound stream.
Existing fixed-batch benchmark wrappers remain unchanged. D16, N1000, B64 and
unmeasured metadata are ineligible for this experimental path.

Cleanup drains the pool before acknowledging release. An interrupted or failed
allocation retains a conservative charge until cleanup; committed bytes are
reported separately. Failed submission poisons the owner. A failed drain does
not clear its charge, and context cleanup preserves an existing forward error.
Released live tensor bytes do not imply that Torch returned cached allocation
to the device allocator.

Validation: all180 Rust workspace tests and47 related Python tests pass,
including7 new core tests,5 bridge tests and13 model-owner cases. The CPU
model-owner tests use explicit allocation/device test doubles around the real
Rust lease; they are not numerical or GPU memory evidence. Core/bridge Clippy
checks pass with warnings denied. Next gates are mixed-batch GPU score/PV
bitwise comparison, N1536/N1000 model arrays/CIF, then isolated paired E2E.

### Mixed-batch GPU qualification passed

One `ModelWorkspaceFusion` instance passes18 complete calls on HIP7: seeds
101/102/103, head-last and head-last-transposed bias, B256/B512/B1024 in
alternating ascending/descending order. The reference computes full-batch
native QK and PV with the already accepted in-place HIP softmax. Every
probability/output bit and output stride matches, including fully masked rows
and broadcast masks. Previously returned results remain unchanged.

All42 microchunk kernel calls use the same score pointer and one9GiB allocation.
Cross-thread/stream rejection leaves the lease unchanged. Actual released live
tensor bytes equal9,663,676,416, matching the final Rust receipt. Peak Torch
allocated memory is55,429,834,752 bytes while holding the full reference; this
is not a model-memory measurement. No latency claim comes from this probe.

Evidence: `provenance/openfold3_model_workspace_qualification.json`. The archive
`model-workspace-v1-results.tar.gz` includes source and rebuilt framework
binaries; SHA256 `678f049127c3502a0b04807b93b2fdc38e39e07de29ca86c85e358fc840dd4d9`.

### Opt-in frozen-worker overlay

`workspace_dispatch.py` composes a separate Rust operator registry around the
accepted `AttentionDispatcher`; it does not replace its D16 or fallback policy.
`model-workspace.patch` applies only to the frozen head16 worker whose SHA256 is
`2ddc92f55f70e3b25b38e721fa71eb3f91a72720b0bf386f36fbfef6a357216e`.
Apply it without fuzz to a fresh copy of that deployment. Both comparison arms
must use the same patched worker and rebuilt framework binaries.

Set `FIREPROT_QK_WORKSPACE` explicitly to `0` or `1`. The enabled arm loads the
controls at `FIREPROT_QK_WORKSPACE_CONTROL` only after checking
`FIREPROT_QK_WORKSPACE_CONTROL_SHA256`. Its context owns the workspace until
forward completes or raises; runtime reporting occurs after cleanup. The report
contains both registries, fallback reasons, workspace accounting, original cache
and head-dimension counters, chunk tuner choices and Torch memory peaks.

All51 related CPU tests pass, including cached selection with dynamic invocation
failure, eligibility changes, untouched disabled dispatch and control-hash
rejection. The initial test run caught a missing required `inplace=False` signature
argument; the corrected code passed before deployment. Model acceptance and E2E
performance remain separate gates. Concurrent correctness runs must not be used
as isolated latency evidence.

### Model correctness passed; coverage changed with the tuner

The v1 model gates complete successfully at N1000 and N1536 with the unchanged
FP32 strict deterministic R3/D200/S1 seed101 contract. At N1536 all11 arrays
match both the accepted D16 baseline and original native anchor bitwise; complete
CIF bytes also match. N1000 matches its original native anchor and allocates no
score workspace. These two correctness runs overlapped on HIP6/HIP7; their
durations are not isolated speedup measurements.

The N1536 result exposes a coverage problem. MSA and Pairformer cached chunks
change from1024 to512; other recorded tuners match the accepted baseline.
The workspace runs only28 times (B256:18, B512:8, B1024:2), totaling42 kernel
submissions. Meanwhile the accepted fallback records5,040 B128 calls. The
9GiB workspace is released successfully, but retaining that much memory may
have influenced tuning. Peak Torch allocated memory is51,757,716,992 bytes.

The source-pinned upstream `ChunkSizeTuner` tests candidate sizes and catches
all `RuntimeError` exceptions. A changed cached chunk therefore does not prove
an OOM without the rejected-call error. Before paired timing, revise the pool
capacity/B128 coverage and capture the tuner rejection causes. Do not force
the old chunk setting or infer a speedup from the earlier operator screen.
No paired timing was launched for v1.

Evidence: `provenance/openfold3_model_workspace_model_gates.json`; archive
`workspace-model-e2e-v1-gates.tar.gz`, SHA256
`f27f5e4881cc5a63b2959e30dec3ba5c0a025ce47337d00d5a18742d67519393`.

### Scoped tuner diagnostics

`tuner_observer.py` records candidate chunk sizes and exact RuntimeError types/
messages while rethrowing the same exception into the unchanged upstream tuner.
Cached calls do not execute extra trials. The source hash and method origin are
checked before installation; context/ownership guards and restoration are tested.
Enable it explicitly with `FIREPROT_CHUNK_DIAGNOSTICS=1` in the experimental
worker overlay; keep it disabled for final latency measurements.

The existing confidence-cap scope must enter before the class observer. Otherwise
its source guard correctly rejects the replaced tuner method. A first diagnostic
run terminated before any pool allocation for this integration error; a regression
test reproduces it and verifies the corrected ordering. Confidence retains its
captured instance callable and its existing report; the observer records the
remaining class-dispatched tuners, including MSA and Pairformer. Ten observer/
confidence-policy tests pass. Model diagnostic acceptance remains required.
