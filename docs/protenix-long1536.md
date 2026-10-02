# Real 1,536-residue inference and CK attention diagnosis

Status: experimental, 2026-10-03. This bucket is opt-in. Enabling
`FIREPROT_ENABLE_LONG1536=1` selects SDPA attention; additionally setting
`FIREPROT_ENABLE_CK_ATTENTION_1536=1` selects the slower CK candidate for
experiments. The validated N1000 default is unchanged.

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
not a controlled warm end-to-end speedup claim.** Same-source repeated
performance runs are still required. The source, allocator, stage timings,
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
attention reference. A full-model native-agreement run of **padded CK** has
not started: DiamondHill SSH became unreachable after the operator tests.
The successful full-model comparison above covers SDPA only.

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

Next work is to compare controlled warm model runs, optimize the still-native
N1536 triangle multiplication and pair epilogues, and investigate layouts
and scheduling that can beat SDPA before promoting a new CK default.
The SDPA validation recorded 432 accelerated attention calls but 144 native
triangle multiplications, 144 native gated transitions, and 3,973 native
LayerNorm calls. Its accumulated triangle-multiplication stage was 33.489 s
versus 20.322 s for attention. These single-call stage observations prioritize
the multiplication path for the next end-to-end optimization; nested stage
timings must not be summed as disjoint model time.
