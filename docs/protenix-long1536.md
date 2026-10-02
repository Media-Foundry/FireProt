# Real 1,536-residue inference and CK attention diagnosis

Status: experimental, 2026-10-03. This bucket is opt-in. Enabling
`FIREPROT_ENABLE_LONG1536=1` selects SDPA attention; additionally setting
`FIREPROT_ENABLE_CK_ATTENTION_1536=1` selects the slower CK candidate for
experiments. The validated N1000 default is unchanged.

The same length opt-in now also enables packed triangle multiplication.
Set `FIREPROT_DISABLE_LONG_TRIANGLE_PACKED=1` for the SDPA-only comparison.

## Controlled warm model comparison

Source `8f31220ccb88dc05ed3e4a3e3c6df64fe791546a` was measured sequentially
on physical GPU 7 with GPU 6 idle, one warmup and three measured calls per
configuration. All runs used the workload below and expandable segments.
Continuous process isolation checks passed. The run order was native,
SDPA-only, then SDPA plus packed multiplication.

| Median GPU time | Native | SDPA only | SDPA + packed multiplication |
| --- | ---: | ---: | ---: |
| Model | 137.709 s | 62.396 s | 41.771 s |
| Triangle attention | 94.382 s | 20.238 s | 20.225 s |
| Triangle multiplication | 34.570 s | 33.420 s | 12.861 s |

Overall model speedup is **3.297x**; packed multiplication adds **1.494x**
over SDPA-only. Stage boundaries overlap and must not be summed. Allocated
peak falls from 53,379,450,880 to 23,929,145,344 bytes. Packed multiplication
does not raise that full-model peak relative to SDPA-only, even though its
isolated operator uses more workspace. Complete worker wall time falls from
611.173 to 227.207 s; that includes one warmup and three measured calls,
not one cold request. Three repetitions and a fixed run order limit the
precision of these comparisons.

Packed multiplication also passed native agreement over 12,139 atoms:
aligned RMSD `0.00012370065 A`, maximum coordinate component delta
`0.0009765625 A`, and maximum confidence delta `0.00003814697`.
The packed kernels are handwritten HIP plus library BMM, not CK GEMM.
See [model evidence](../provenance/protenix_long1536_packed_8f31220.json).

## Workload and native agreement

The input `umem_L01536_6b140cc19f02d9b9` is a complete 1,536-residue sequence
from the existing local OpenFold UniProt membrane benchmark corpus. It was
not cropped, repeated, concatenated, or padded. The input metadata records
the source FASTA and sequence hashes. It has no experimental-coordinate
mapping and does not establish structural accuracy.

The exact workload uses Protenix Mini-default v0.5.0, FP32, four recycle
cycles, one diffusion step, one sample, seed 101, hipBLAS, TF32 disabled,
and no external MSA or template search. Upstream chooses an outer chunk of
512 at N1536. Each triangle attention therefore invokes three operations
with QKV shape `[512,4,1536,32]`, strides `[196608,32,128,1]`, and a broadcast
bias `[1,4,1536,1536]`. Treating the outer batch as 1536 would benchmark a
different execution path.

The native validation initially failed with the default allocator while
trying to allocate an 18 GiB attention buffer. A separate allocation probe
confirmed that this ROCm build supports expandable segments. Both successful
native and FireProt validations used
`PYTORCH_ALLOC_CONF=expandable_segments:True` on physical GPU 7. The native
model semantics and upstream chunk size were preserved.

At source `2cf3ef50e67dae466021568cfd4482c6091150c6`, SDPA passed the locked
`1e-3` native-agreement gates across 12,139 atoms:

- Aligned coordinate RMSD: `1.10854495e-4 A`.
- Maximum coordinate component delta: `9.765625e-4 A`.
- Maximum confidence delta: `2.67028809e-5`.

These were zero-warmup, one-call validation runs. Their model times were
142.958 s native and 67.972 s FireProt, and their allocated peaks were
52,130,649,600 and 22,680,344,064 bytes. **These are validation observations,
not a controlled warm end-to-end speedup claim.** The subsequent controlled
comparison is recorded above. The source, allocator, stage timings,
selection counts, and comparison are retained in
[validation evidence](../provenance/protenix_long1536_validation_2cf3ef5.json).

## What the SDPA comparison measures

SDPA dispatches AOTriton `attn_fwd`. Both it and the current CK Tile kernel
fuse QK, softmax, and PV. The current QRKSVS pipeline already caches Q and
normalizes the output after the key-block loop; the historical generic CK
pipeline's repeated Q loads are not an explanation for this kernel.

The canonical operator comparison uses physical GPU 6, random seed 419,
FP32, TF32 disabled, the observed head-last QKV and bias storage, five
warmups, and twenty measured calls. It includes bias packing and output
allocation; compilation is excluded. Continuous process monitoring passed.

| Candidate | Median ms | Interpretation |
| --- | ---: | --- |
| Direct SDPA | 34.83–34.88 | Reference implementation |
| CK M128/N64/W4 | 48.42 | N1000 configuration regresses at N1536 |
| CK query-major grid | 48.10 | Grid remapping alone does not close the gap |
| CK M64/N64/W2 | 45.02 | Best of four initial tile/warp alternatives |
| CK M128/N64/W4, bias row +16 | 39.39 | Retained experimental improvement |
| CK MFMA32, corrected distribution | 87.54 | Correct but rejected for performance |
| CK MFMA32, corrected distribution, bias row +16 | 41.71 | Still slower than retained candidate |

The bias change allocates physical storage `[1,4,1536,1552]` and copies
the logical bias into its first 1536 columns. The HIP kernel uses row stride
1552 and head stride `1536*1552`, while both logical sequence lengths remain
1536. The extra columns are not residues and do not enter the softmax.
The experiment isolates address layout without changing matrix tiles,
precision, or attention semantics. It cuts CK operator time by about 18.6%.

The integrated Python/HIP path subsequently measured **39.415 ms median,
39.549 ms p95**, versus direct SDPA **34.838/34.929 ms** and the final SDPA
repeat **34.829/34.900 ms**. Its maximum absolute operator delta was
`5.96046448e-6`; the `atol=rtol=2e-5` gate passed. Because CK remains slower,
it retains its independent opt-in and is not promoted for this bucket.
Local regression tests passed (100 passed, 14 skipped), as did GPU 6 tests
(113 passed, 1 skipped), including the full B512/N1536 explicit Torch
attention reference. After a temporary DiamondHill SSH outage recovered,
**padded CK also passed full-model native agreement** at source
`475d9f7a548420f060b8ee4eaf9f8a2da62c7eab`, using the same allocator,
model, input, dtype, cycles, steps, sample count, and random seed. The run
selected the N1536 CK implementation 432 times and passed continuous
GPU 7 process monitoring. Across 12,139 atoms, aligned RMSD was
`1.05254271e-4 A`, maximum coordinate component delta was `9.765625e-4 A`,
and maximum confidence delta was `3.81469727e-6`. Its single validation
model call took 69.848 s; it is not a warm performance comparison.
See [padded CK model evidence](../provenance/protenix_long1536_ck_validation_475d9f7.json).

## Counter evidence and rejected hypotheses

[Counter evidence](../provenance/protenix_attention_counters_2cf3ef5.json)
records raw artifact hashes, inputs, libraries, resources, and all three
dispatch samples. Counter runs are diagnostics, not latency benchmarks;
the profiler entry has no continuous external-process monitor. GPU 7 tests
overlapped SQ counter collection. Formal operator timings above were
collected separately.

- A single-call torch profile located the difference in the main kernel:
  approximately 34.61 ms for AOTriton versus 48.23 ms for CK. Bias copies
  were about 0.16 ms on each path.
- Before padding, both paths issued similar external read-request counts
  (about 338 million SDPA versus 347 million CK on a warm dispatch).
  This does not establish identical memory latency or access distribution.
- Bias row padding reduced CK's warm external read requests to about
  303 million. This is supporting counter evidence for the layout change;
  it does not identify the exact hardware conflict mechanism.
- CK issued about 302 million MFMA instructions versus SDPA's 157 million,
  but an instruction-count ratio is not a FLOP ratio: instruction shapes
  perform different amounts of work.
- CK's measured LDS instruction and wait counts were lower. AOTriton's
  zero static LDS allocation did **not** mean zero LDS-class instructions.
- Installed CK lacks the FP32 transposed MFMA32 dispatcher combinations
  needed by this attention pipeline. An isolated adapter using SwizzleB
  compiled but failed numerically (maximum error about 2.21). Removing that
  distribution mismatch restored correctness but did not improve speed.
  Neither adapter is registered in model execution.

The bias-stride ablation establishes that address layout contributes to the
regression. It does not yet identify a particular cache-set, memory-channel,
or bank conflict as the cause. Avoid replacing that remaining uncertainty
with an architectural claim.

Operator experiments, unsuccessful candidates, and two diagnostic C++
sources verified against their measured hashes are retained in
[diagnostic evidence](../provenance/protenix_attention_diagnosis_2cf3ef5.json).

## Reproduce

```bash
# Complete operator including packing and allocation.
HIP_VISIBLE_DEVICES=6 PYTHONPATH=.:Protenix python -m benchmarks.protenix.benchmark_attention \
  --length 1536 --include-ck-tile --output outputs/benchmarks/attention1536.json

# Native-agreement validation. Use a separate directory for every run.
HIP_VISIBLE_DEVICES=7 PYTORCH_ALLOC_CONF=expandable_segments:True \
FIREPROT_ENABLE_LONG1536=1 PYTHONPATH=.:Protenix \
python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250-long1536-validation.json \
  --backend fireprot --blas-backend hipblas \
  --output-dir outputs/benchmarks/long1536-validation
```

Set `FIREPROT_SOURCE_COMMIT` and `FIREPROT_PROTENIX_COMMIT` to the verified
deployed source commits when the remote checkout's Git metadata is not
synchronized. For the native reference use `--backend native` with the same
allocator. For the CK candidate additionally set
`FIREPROT_ENABLE_CK_ATTENTION_1536=1`.

`probe_attention_kernel.py` isolates prepacked attention for rocprof. Its
`--library` must be a prebuilt C ABI library matching the selected length
and `--bias-padding`; the current CK N1536 library requires
`--bias-padding 16`. Check device isolation externally. Use the canonical
benchmark, not profiler-injected durations, for performance decisions.

The preceding SDPA-only validation recorded 432 accelerated attention calls but 144 native
triangle multiplications, 144 native gated transitions, and 3,973 native
LayerNorm calls. Its accumulated triangle-multiplication stage was 33.489 s
versus 20.322 s for attention. These single-call stage observations prioritize
the multiplication path for the next end-to-end optimization; nested stage
timings must not be summed as disjoint model time. Packed multiplication
has since addressed that priority in the controlled comparison above.

## Why SDPA still beats CK: additional ablations

SDPA is an API; the measured implementation is AOTriton's fused attention
kernel. The installed source caches Q outside the key loop, uses online
softmax, and normalizes the output at the end. The current CK QRKSVS
pipeline already does these things. The older generic CK explanations
(reloading Q and repeatedly normalizing the output) do not describe the
current candidate. Upstream describes the algorithm in its
[forward kernel](https://github.com/ROCm/aotriton/blob/main/modules/flash/kernel/fwd_kernel.py);
the version inspected here remains the pinned 0.11.2 build, not main.

The new experiments preserve the same arithmetic and change scheduling
or storage. All used GPU 6 alone, FP32, seed 419, five warmups and twenty
measured calls. Packing and required output conversion are included.

| Complete attention operator | Median ms | Result |
| --- | ---: | --- |
| SDPA, before/after | 34.89 / 34.82 | Reference |
| Padded CK M128/N64/W4 | 39.41 | Current experimental path |
| Same CK, batch group 8 | 37.93 | 3.8% lower latency; still slower than SDPA |
| Same CK, batch group 32 / 512 | 38.23 / 38.58 | Larger groups did not help |
| CK M64/N64/W2, batch group 8 | 38.54 | Smaller tiles did not improve the best result |
| SDPA with batch/head exchange | 36.62 | Rejected |
| SDPA with contiguous QKV, including packing | 36.64 | Rejected |
| CK with contiguous QKV, best tested tile/group | 38.64 | Rejected against 37.93 ms |

Each group runs neighboring blocks over different outer batches sharing
the same bias tile. This scheduling ablation improves CK, but does not
prove a particular cache mechanism. All candidates passed the operator
`atol=rtol=2e-5` gate; CK maximum absolute delta was `5.96046448e-6`.
No new candidate has a model-level promotion or E2E speedup claim.
[Experiment evidence](../provenance/protenix_attention_scheduling_8f31220.json)
retains build commands, source/library hashes, failed alternatives and
continuous isolation checks.

The remaining explanation is narrower than "CK needs fusion": the measured
gap is inside already-fused kernels. Address layout is a demonstrated
contributor; thread scheduling adds a smaller benefit. Matrix instruction
mapping and the overlap of memory loads, softmax and matrix work remain
priorities for controlled experiments. Earlier counters show different
MFMA counts, but different instruction shapes preclude interpreting the
ratio as twice the arithmetic. Lower CK LDS counts also rule out treating
LDS volume alone as the explanation. Those counters were diagnostic runs
with a weaker isolation protocol than the latency benchmarks.

Next work should compare actual ISA and dependency chains, then implement
a compatible FP32 MFMA32 distribution or improve load/compute overlap.
Changing only the MFMA type is insufficient: earlier CK distribution
changes failed numerically or regressed. Keep the existing FP32 semantics,
four recycle cycles and numerical gates. Since attention still accounts
for about 20.2 s of the 41.8 s model, stage-level improvements remain a
major target; an operator percentage is not a model percentage.

Reproduce the standalone scheduling study with the measured CK headers:

```bash
HIP_VISIBLE_DEVICES=6 PYTHONPATH=.:Protenix python -m benchmarks.protenix.build_attention_schedule \
  --output-dir runtime/ck_schedule
HIP_VISIBLE_DEVICES=6 PYTHONPATH=.:Protenix python -m benchmarks.protenix.benchmark_attention_schedule \
  --library-dir runtime/ck_schedule --output outputs/benchmarks/ck_schedule.json
```

Use `--tile 64` with the builder for the two-wave variant. For contiguous
QKV use `--contiguous-qkv` with both commands and a distinct output directory.
The builder modifies a private header overlay and never the ROCm installation.

### Pipeline follow-up

With M128/N64/W4 and batch group 8 held fixed, moving V prefetch until after
the QK tail measured 38.35 ms; delaying it until after softmax measured
40.72 ms. Enabling the CK fast-exp2 path measured 39.96 ms. All passed the
operator gate, but none beat the preceding 37.95 ms configuration. SDPA
before/after measured 34.88/34.82 ms. The installed whole-K prefetch policy
rejected both M64 and M128 FP32 candidates at compile time with
`Not supported data types!`; no runtime result is claimed for those builds.
See [pipeline evidence](../provenance/protenix_attention_prefetch_8ddbdfb.json).

### Pair epilogue candidate

The length opt-in now also enables the existing HIP LayerNorm and gated
transition for the exact 1536-by-1536 pair bucket. Set
`FIREPROT_ENABLE_LONG1536_EPILOGUES=0` to disable them for comparisons.
The bucket remains opt-in and preserves FP32 and C4.
Random-weight, full-operation measurements on GPU 6 (seed 641, five warmups,
twenty repetitions) measured LayerNorm at 11.729 ms native versus 1.931 ms
HIP, and gated transition at 63.811 versus 46.657 ms. Native repeats were
11.736 and 63.557 ms. Numerical gates and continuous isolation checks passed.
Regression tests passed locally (102 passed, 20 skipped) and on GPU 6
(121 passed, 1 skipped). These are operator results, not model speedups.

At source `547880cb5d225215a0b6a7018099509aaf7ca2fa`, a same-source model
comparison on GPU 7 (GPU 6 idle, one warmup, three measured calls, fixed
disabled/enabled order) measured **41.444 s disabled versus 36.664 s enabled**:
**1.130x** speedup, or **11.53%** less model time. Nearest-rank p95 was
41.711 versus 36.744 s. Peak allocated memory decreased from 23,929,145,344
to 22,721,185,792 bytes. Both continuous process monitors passed.

The model agreement run passed over 12,139 atoms: aligned RMSD
`0.00012953505 A`, maximum coordinate component delta `0.0009765625 A`,
and confidence maximum delta `0.00000381470`. It selected 438 accelerated
LayerNorm calls and 72 accelerated transitions; unsupported shapes continued
to use native code. The attention-stage median also drops from 20.228 to
18.829 s because its surrounding LayerNorm work is included in that stage;
the SDPA kernel itself is unchanged. See
[epilogue model evidence](../provenance/protenix_long1536_epilogues_547880c.json).

An isolated FP32 adaptation of the whole-K policy subsequently compiled
after adding FP32 dispatch, a no-sink compatibility declaration, K1=32,
and K-pack=2 for M128. Both candidates passed the operator gate (maximum
absolute delta `1.37090683e-6`) but measured 42.80 ms for M64 and 40.72 ms
for M128, versus SDPA's 34.86/34.80 ms. They remain excluded from model
dispatch. This rejects the tested whole-K configurations, not all possible
FP32 prefetch designs.

### Bias row-stride sweep with fixed batch scheduling

A subsequent complete-operator sweep held M128/N64/W4 and batch group 8
fixed and varied only the physical bias row stride. All candidates used
one binary with a runtime stride argument; logical sequence length stayed
1536. On GPU 6, seed 419, FP32, five warmups and twenty measured calls:

| Extra elements per bias row | Median ms |
| --- | ---: |
| 0 | 44.447 |
| 4 | 37.830 |
| 8 | 37.702 |
| 16 | 37.955 |
| 32 | 38.047 |
| 48 | 37.946 |
| 64 | 43.950 |
| 96 | 38.048 |
| 128 | 43.771 |
| 256 | 43.083 |

SDPA before/after measured 34.888/34.826 ms. Every CK candidate passed the
operator gate with maximum absolute delta `5.96046448e-6`; continuous GPU
isolation passed. Bias packing and output allocation are included. The
best tested stride is only about 0.7% faster than +16 and still slower than
SDPA, so no model dispatch change follows this sweep. The regressions at
some larger strides establish address-layout sensitivity, but do not prove
a particular cache-set or memory-channel conflict. The fixed test order
and single seed further limit interpretation of small differences.
See [stride evidence](../provenance/protenix_attention_bias_stride_fb06beb.json).

```bash
HIP_VISIBLE_DEVICES=6 PYTHONPATH=.:Protenix python -m benchmarks.protenix.benchmark_attention_bias_stride \
  --schedule-dir runtime/ck_schedule --output outputs/benchmarks/ck_bias_stride.json
```

The schedule directory must first be produced by the scheduling builder
above with its default BSHD layout. This scan does not register a kernel.

### Mixed MFMA32 mappings require an explicit distribution contract

The next study separated the two GEMMs: QK-only MFMA32, PV-only MFMA32,
and MFMA32 in both. It retained M128/N64/W4, bias padding 16, batch group 8,
FP32 and the same operator measurement protocol. The prior numerically
correct, unswizzled transposed MFMA32 adapter was used.

Both one-sided substitutions failed the numerical gate by large margins
(maximum absolute errors about 1399.68 and 30.07). They were rejected before
timing. The installed gfx90a pipeline passes the softmax result directly
to PV, and indexes row statistics using the output accumulator's distributed
spans. It therefore relies on compatible QK/PV thread distributions.
Changing only the instruction type does not preserve that contract.

A private pipeline overlay converted QK logits through LDS into a distribution
constructed from the PV warp shape, before softmax. It also derived the
softmax row-statistic type from that converted distribution. This restored
operator correctness for both mixed variants, supporting the distribution
diagnosis. All converted candidates had maximum absolute delta at most
`6.07967377e-6`, but their complete-operator timings regressed:

| QK / PV mapping | Without conversion | With LDS conversion |
| --- | ---: | ---: |
| MFMA32 / MFMA16 | Numerical failure | 49.211 ms |
| MFMA16 / MFMA32 | Numerical failure | 52.376 ms |
| MFMA32 / MFMA32 | 40.204 ms | 51.997 ms |

SDPA measured 34.83–34.89 ms. The conversion adds an LDS round trip and
barriers, and raises the requested shared-memory footprint to at least
32 KiB per block. It is a correctness reference, not a promoted optimization.
A future mixed mapping needs a cheaper lane/register redistribution and
must preserve both P and row-statistic distributions. No model path changed.
The [mixed-mapping evidence](../provenance/protenix_attention_mixed_mfma_6d42e2f.json)
includes exact diagnostic C++ source, pipeline replacement text, compile
commands, numerical failures, timings and continuous GPU 6 isolation checks.

This study also found that failed `torch.testing.assert_close` diagnostics
could retain tensors through traceback cycles. The standalone scheduling
benchmark now releases candidate outputs and collects cycles before the
next candidate. A repeat with the same two failures restored the subsequent
CK allocation peak from 3,095,789,568 to 2,089,156,608 bytes and final SDPA
peak from 3,498,049,536 to 2,491,416,576 bytes. The earlier affected peak
records are retained with an explicit limitation; they are not model memory
claims. This cleanup is outside the timed GPU interval.

### Wave-local conversion and a four-wave mapping proof

A CPU coordinate exporter now evaluates CK's own distributed-coordinate
functions for every thread-buffer element. For the measured M128/N64/W4
mixed mappings, both directions form complete 8,192-element bijections,
but **6,144 elements (75%) cross wave boundaries**. A wave shuffle alone
cannot implement those mappings.

For M64/N64/W1, each direction forms a complete 4,096-element bijection
within one wave. Affine lane/register expressions were checked against all
64 lanes, then used to generate 128 shuffle calls per conversion. Every
source-register shuffle executes before the destination selects its value,
avoiding reads from inactive lanes. These candidates passed the GPU operator
gate without the LDS conversion, but were too slow:

| M64/N64/W1 candidate | Median ms |
| --- | ---: |
| MFMA32 QK / MFMA16 PV, wave shuffle | 59.837 |
| MFMA16 QK / MFMA32 PV, wave shuffle | 55.483 |
| MFMA16 / MFMA16 control, no conversion | 61.677 |
| SDPA before / after | 34.909 / 34.832 |

The improvement relative to the single-wave control does not justify
promotion over the existing multi-wave path or SDPA. All numerical gates
passed (maximum absolute delta at most `6.07967377e-6`), as did continuous
GPU 6 isolation. The test retained the full B512/N1536 shape, FP32, seed 419,
five warmups, twenty measured calls, bias packing and output allocation.

A further CPU proof removes cross-wave dependencies while retaining four
waves by permuting rows inside each 128-row query block. For QK32/PV16,
the local row permutation is:

```text
p(m) = (m & 15) | ((m & 16) << 2) | ((m & 96) >> 1)
```

For QK16/PV32, use its inverse:

```text
p_inverse(m) = (m & 15) | ((m & 64) >> 2) | ((m & 48) << 1)
```

Both retain all 8,192 logical tile elements and reduce the verified
cross-wave count to zero. **This is only a mapping proof.** A corresponding
GPU kernel must apply the permutation consistently to Q and bias row loads,
then convert logits into the PV-compatible distribution before softmax.
K/V and the logical output order must retain their original semantics.
At that point the kernel had not been tested. The subsequent GPU study
below implements and measures these permutations.

[Wave-conversion evidence](../provenance/protenix_attention_wave_shuffle_a8d7e33.json)
records the generated conversion, compiler commands, mappings, library
hashes and GPU results. The standalone CPU mapping tool can be reproduced
against the measured CK installation:

```bash
/opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx90a -I/opt/rocm/include \
  -DBM=64 -DWM=1 -DG0M=32 -DG0K=8 -DG1M=16 -DG1K=16 \
  benchmarks/protenix/csrc/ck_attention_mapping.cpp -o /tmp/ck_attention_mapping
HIP_VISIBLE_DEVICES=6 LD_LIBRARY_PATH=/opt/rocm/lib /tmp/ck_attention_mapping > /tmp/ck_mapping.txt
```

The first `WM*64` lines describe QK coordinates, followed by the same number
of lines for the PV-compatible softmax distribution. Each integer encodes
`row*64 + column` in thread-buffer order. This executable launches no GPU
kernels. Use BM=128/WM=4 for the four-wave mapping, or exchange the G0/G1
macro pairs for the reverse direction.

### Four-wave row-permuted GPU implementation

The mapping proof was implemented in private CK pipeline overlays. Q and
bias tensor views are split into 128-row blocks and transformed by swapping
their middle row-index dimensions. K/V retain their original addressing.
The converted logits and softmax row statistics use the PV-compatible
distribution, preserving the logical output order. The diagnostic C ABI
rejects lengths other than B512/N1536; this is not a generic masked kernel.

The first implementation generated 64 wave shuffles. A second decomposed
the full 11-bit lane/register permutation into adjacent bit swaps: one
lane-bit swap, one lane/register-bit exchange, and register-only reindexing.
Composition was checked for all 2,048 elements of each wave. This reduces
the generated shuffle count to 48 without an LDS round trip or additional
block synchronization.

| M128/N64/W4, bias padding 16, batch group 8 | 64 shuffles | 48 shuffles |
| --- | ---: | ---: |
| MFMA32 QK / MFMA16 PV | 39.336 ms | 39.255 ms |
| MFMA16 QK / MFMA32 PV | 38.813 ms | 38.659 ms |
| SDPA, final control | 34.830 ms | 34.804 ms |

Both implementations passed the full operator gate (maximum absolute delta
at most `6.07967377e-6`) and continuous GPU 6 isolation checks. The protocol
retained FP32, seed 419, five warmups, twenty measured calls, bias packing
and output allocation. The best result remains slower than SDPA and the
preceding MFMA16 CK batch-group configuration. No model path is changed,
and no incremental model speedup is claimed. These results substantially
reduce the cost of the earlier LDS conversion but do not justify promotion.
See [GPU row-permutation evidence](../provenance/protenix_attention_row_permute_878074e.json).

Further attention tuning now has a measured diminishing return. The next
operator priority is the packed triangle contraction: outgoing calls use
`bmm(a, b.transpose(-1, -2))`, incoming calls use
`bmm(a.transpose(-1, -2), b)`, with contiguous physical channel-first
projections. At N1536 the logical batches are `[128,1536,1536]`. The existing
dimension-only BLAS replay does not preserve these transpose strides, so
these exact contractions need their own CK/BLAS comparison before choosing
a replacement.

### Explicit CK packed BMM comparison

The actual outgoing NT and incoming TN layouts were replayed with physical
contiguous `[128,1536,1536]` projections, seed 443, FP32, TF32 disabled,
five warmups and twenty measured calls. Every candidate was checked against
all elements of saved hipBLAS outputs before timing. GPU 6 isolation checks
passed for every run; output allocation is included.

| Implementation | Outgoing median ms | Incoming median ms |
| --- | ---: | ---: |
| hipBLAS | 29.526 | 29.414 |
| Process CK preference | 29.601 | 29.491 |
| hipBLASLt | 25.916 | 25.732 |
| hipBLAS repeat | 29.617 | 29.467 |
| Explicit CK MFMA32 | 36.690 | 33.952 |
| Explicit CK MFMA16 | 39.549 | 35.963 |

The installed PyTorch `bgemm_internal<float>` source branches only on
Cublaslt preference; all other preferences fall through to ordinary BLAS.
Thus the process CK row does not demonstrate a CK BMM kernel. The new C ABI
probe explicitly instantiates `DeviceBatchedGemmXdl` with a 128x128 tile,
256 threads and K=32, using the two actual transpose layouts. Both MFMA
variants pass the full numerical gate (maximum delta below `2e-7`), but
both are slower than hipBLAS. Compiler warnings report achieved occupancy
below the requested target; that is a tuning lead, not a proven cause.
Neither candidate is registered in model dispatch.

hipBLASLt reduces this isolated contraction's time by approximately 12–13%.
No N1536 model improvement is established by this result; changing global
BLAS policy also changes other projections. The next gate is a controlled
model comparison or an explicit per-operation Lt path. Full reports,
source hashes, compiler logs and the PyTorch dispatch finding are retained
in [BMM evidence](../provenance/protenix_packed_bmm_e6dfa14.json).

Reproduce the explicit candidate after compiling the probe into a shared
library with the recorded flags:

```bash
HIP_VISIBLE_DEVICES=6 PYTHONPATH=.:Protenix python -m benchmarks.protenix.benchmark_packed_bmm \
  --backend direct_ck --library runtime/ck_packed_bmm_probe/mfma32.so \
  --reference-dir outputs/benchmarks/packed-bmm1536-e6dfa14/reference \
  --output outputs/benchmarks/packed-bmm1536-direct-ck.json
```

Create the full reference first with `--backend hipblas --write-reference`
and the same reference directory, omitting `--library`. Each process fixes
its BLAS policy once. References and output files stay outside Git.

### Global hipBLASLt does not improve the complete N1536 model

A same-source follow-up retained the current SDPA, packed multiplication,
and HIP epilogues, changing only the process BLAS preference. Source
`e6dfa14f47150678450c573f5d793e71d8f5910f`, GPU 7 with GPU 6 idle,
Mini-default FP32 C4/S1, seed 101, sample 1, one warmup and three measured
calls were used. Both runs dumped predictions outside the GPU model interval.
Continuous isolation checks passed. The fixed run order was hipBLAS then Lt.

| Metric | hipBLAS | hipBLASLt |
| --- | ---: | ---: |
| Model median | 36.481 s | 36.788 s |
| Model p95 | 36.900 s | 37.282 s |
| Triangle multiplication median | 11.168 s | 11.252 s |
| Triangle attention median | 18.804 s | 19.106 s |
| Peak allocated memory | 22,721,185,792 B | 22,800,877,568 B |

The model speedup is 0.992x: no observed benefit from global Lt. Three
samples in a fixed order do not establish a small regression statistically.
Both runs selected SDPA 1,728 times and packed triangle multiplication 576
times across warmup and measured calls. All three same-seed output pairs
passed the locked coordinate/confidence gates over 12,139 atoms each.
The isolated BMM advantage therefore does not justify a global switch.
A per-operation Lt implementation or real-call kernel trace is needed to
separate contraction improvements from projection changes and runtime dispatch.

Before this comparison, a checksum audit detected one stale remote HIP
source file; it was restored to the committed version before either run.
Model-source and benchmark checksums were then consistent, excluding
hardware-local generated LayerNorm build artifacts. This audit does not
retroactively establish the source state of earlier runs. See
[complete model evidence](../provenance/protenix_long1536_blas_e6dfa14.json).

### CK BMM tile and prefetch tuning

A follow-up kept FP32 MFMA32 and 256 threads, while varying tile size,
K-block depth and prefetch stages. The same full-output references,
seed 443, GPU 6, five warmups and twenty repetitions were retained.
All executable candidates passed numerical and continuous isolation gates.

| CK configuration | Outgoing median ms | Incoming median ms |
| --- | ---: | ---: |
| 64x64, K32, one prefetch stage | 30.268 | 29.777 |
| 64x64, K64, one prefetch stage | 43.098 | 45.787 |
| 128x128, K32, two prefetch stages | 41.484 | 37.354 |
| hipBLAS, final control | 29.591 | 29.481 |

The 64x64 K32 candidate improves on the earlier explicit CK 128x128
configuration (36.690/33.952 ms), but remains slower than hipBLAS in both
directions. Smaller incoming tiles need a different transfer cluster to
retain vectorized loading. This candidate emits no occupancy warning;
the K64 configuration does. These compiler observations do not establish
achieved runtime occupancy or isolate the cause of the timing difference.
The 128x128 K64 candidate fails compilation because its 66,016-byte LDS
request exceeds the 65,536-byte limit. No runtime result is assigned to it.

The probe now exposes `TILE`, `FIREPROT_K0` (K-block divided by four), and
`PREFETCH` compile-time options, preserving the original default instance.
An initial `K0` macro collided with CK header identifiers and was renamed;
an initial 64-tile transfer cluster violated vector-width constraints and
was corrected before the results above. No model dispatch changed. The
next useful candidate is a different writeback/pipeline implementation,
rather than promoting a contraction that still loses to the baseline.
See [tuning evidence](../provenance/protenix_packed_bmm_tuning_5b6c5bd.json).

### CShuffle pipeline comparison

An explicit `DeviceBatchedGemmMultiD_Xdl_CShuffle_V3` probe now implements
the same FP32 contractions with no D tensors, scalar pass-through epilogues,
and vectorized CShuffle writeback. This is a separate diagnostic C ABI;
it is not registered in model execution. The actual NT/TN layouts,
full hipBLAS reference checks, GPU 6 isolation, seed 443, five warmups and
twenty repetitions were preserved. All candidates compiled and passed.

| Tile / pipeline | Outgoing ms | Incoming ms |
| --- | ---: | ---: |
| 64x64 / v1 | 31.229 | 29.724 |
| 128x128 / v1 | 38.500 | 36.200 |
| 64x64 / v3 | 30.533 | 29.077 |
| 64x64 / v4 | 34.705 | 36.422 |
| 64x64 / v5 | 30.597 | 29.115 |
| hipBLAS control | 29.641 | 29.496 |
| hipBLASLt control | 25.918 | 25.728 |
| 64x64 / v3 repeat after controls | 30.558 | 29.073 |

The v3 incoming contraction has a repeatable isolated advantage of about
1.4% over hipBLAS. Outgoing still regresses, and both directions remain
slower than hipBLASLt. These results do not establish model speedup or
justify replacing both contractions. The next priority is to measure an
explicit per-operation Lt path while retaining hipBLAS for other GEMMs,
so the earlier global-policy comparison cannot confound that experiment.
The CShuffle probe retains compile-time `FIREPROT_TILE` and
`FIREPROT_PIPELINE` options (pipeline numbers 1 through 5 map to CK v1–v5).
Full commands, logs, binary hashes, numerical and monitoring results are
in [CShuffle evidence](../provenance/protenix_packed_bmm_cshuffle_0c7ad00.json).

### Explicit per-operation hipBLASLt probe

A standalone C ABI now calls hipBLASLt directly while the PyTorch process
policy remains hipBLAS. It uses FP32 compute, the first heuristic algorithm,
and column-major descriptors with reversed physical operands to implement
the original outgoing NT and incoming TN row-major results. No global
policy mutation is needed. All elements of both outputs passed the saved
hipBLAS reference gate (maximum delta `3.2783e-7` and zero respectively).

On GPU 6, seed 443, TF32 disabled, five warmups and twenty repetitions:

| Implementation | Outgoing median ms | Incoming median ms |
| --- | ---: | ---: |
| Explicit Lt | 25.939 | 25.740 |
| hipBLAS final control | 29.605 | 29.424 |

Both process monitors passed. This preserves the isolated Lt advantage
without changing the process BLAS policy. It does not establish a model
speedup. With 72 calls per direction, multiplying these differences gives
about 0.53 seconds, an estimate that excludes any integration effects.

The diagnostic is explicitly single-device and single-thread. Each
direction owns a persistent 32 MiB `hipMalloc` workspace; their combined
64 MiB is recorded separately and is outside PyTorch's peak allocation.
The apparent PyTorch allocation difference must not be claimed as model
memory savings. Setup and heuristic selection occur before warm timing.
A model implementation needs PyTorch-managed workspace, safe stream/device
ownership, exact capability gates and fallback before full-model testing.

Reproduce using `benchmark_packed_bmm --backend direct_lt --library <so>`
with the existing reference directory. Compile flags, source/library
hashes and full measurements are retained in
[direct Lt evidence](../provenance/protenix_packed_bmm_direct_lt_ed3d91e.json).

### Experimental model route for per-operation Lt

`FIREPROT_ENABLE_LT_BMM_1536=1` now enables an experimental replacement only
inside packed triangle contraction. It additionally requires contiguous
physical `[128,1536,1536]` FP32 inputs on the same gfx90a device, aligned
pointers, disabled TF32, no gradients and no active stream capture.
Unmatched inputs and unavailable extension builds use the original BMM.
Runtime launch failures are surfaced instead of hidden by another kernel.
The process BLAS preference is unchanged.

The caller allocates a 32 MiB PyTorch workspace on the current stream for
each invocation. Lt descriptors are cached by device and direction, with
host initialization and launch serialization; asynchronous operations use
separate workspace allocations. Source, compiler, headers and library
installation identity fingerprint the lazy build. The runtime manifest
records build identity, errors and successful/fallback call counts.

Focused GPU 6 tests passed for both directions on non-default streams,
including immediate allocator reuse of the released workspace. Local
regression tests passed (108 passed, 21 skipped); focused GPU tests passed
(3 passed). The integrated operator measures 25.963/25.772 ms outgoing /
incoming and passes full saved-output checks. Unlike the standalone probe,
its PyTorch peak allocation includes the temporary workspace. The path
remains opt-in pending a same-source full-model comparison.

At candidate `d4ffc2c43d4e11c6854a5e1422aaf4cf2708ac38`, the first controlled
model comparison on GPU 7 (GPU 6 idle) retained Mini-default FP32 C4/S1,
sample 1, seed 101, one warmup and three measured calls. Run order was
Lt disabled then enabled; both runs used hipBLAS process policy and
expandable segments. GPU isolation passed, as did all three coordinate /
confidence comparisons over 12,139 atoms (maximum aligned RMSD
`0.000105095 A`, maximum confidence delta `3.8147e-6`).

| Metric | Lt disabled | Lt enabled |
| --- | ---: | ---: |
| Model median | 36.495 s | 36.337 s |
| Model p95 | 37.041 s | 36.678 s |
| Triangle multiplication median | 11.194 s | 11.041 s |
| Pairformer stack median | 33.590 s | 33.429 s |
| Peak allocated memory | 22,721,185,792 B | 22,721,185,792 B |
| Successful Lt calls, including warmup | 0 | 576 |
| Packed contraction native fallbacks | 576 | 0 |

The observed median reduction is only 0.43% (1.0043x), below the spread
within these runs. It does not establish a stable E2E improvement and is
smaller than the isolated-operator estimate. The route remains opt-in.
The next diagnostic should time the exact contraction boundary in the
model, separating its contribution from surrounding projection and
normalization work. See [runtime/model evidence](../provenance/protenix_long1536_lt_runtime_d4ffc2c.json).
