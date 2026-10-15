# Performance priorities after the 2026-10-10 evidence review

The [README snapshot](../README.md#latest-direct-native-comparison-per-model-and-length)
lists the latest measured direct ratios. This assessment ranks further work from
local experimental evidence. It does not rank model quality, predict a new
speedup, or treat different timing boundaries as a uniform benchmark.

## 1. OpenFold3 has the clearest measured remaining target

OpenBind-0 N1536 retains strict FP32 with deterministic memory filling, R3/four
trunk passes, D200/S1 and seed101. Three fresh AB/BA/AB pairs give complete
invocation medians **950.891 -> 331.970 s (2.864x)** and model-forward medians
**855.152 -> 237.199 s (3.605x)**. All six outputs match all eleven arrays
bitwise and complete CIF bytes. The composition remains experimental.
[Direct evidence](../provenance/openfold3_direct_native_latest_e2e_n1536.json).

The separate instrumented trace attributes **87.922 s / 35.37%** of summed
kernel time to fused QKV. Sampled CU activity is 99.85%, occupancy 98.75%,
MFMA utilization 25.97% and HBM traffic 65.92 GB/s. This supports investigating
execution scheduling, synchronization and on-chip data movement; it does not
prove which one limits performance. Two smaller-block schedules already passed
exactness but regressed trained module latency and were rejected.
[Hardware evidence and sampling limitations](openfold3-hardware.md),
[rejected schedules](../experiments/mfma_qkv/README.md#counter-guided-smaller-block-trials).

Follow-up reviewed on 2026-10-15: the narrow projection passes trained/full-model
exactness but regresses all three E2E pairs (322.475 -> 324.589 s median, 0.993486x).
It is not promoted. [Paired evidence](../provenance/openfold3_narrow_projection_e2e_20261015.json).

PairFormer-only BF16 completes, but aligned all-atom RMSD versus FP32 is 5.293 A;
unprotected FP16 produces NaN in the first triangle update. Protecting complete
triangle updates in FP32 eliminates the observed nonfinite failure, but aligned
all-atom RMSD is 6.423 A. None of these precision candidates is promoted.
These are instrumented diagnostics, not new speedup results. The input has low
mean pLDDT (~37.9); prediction disagreement alone does not establish experimental
accuracy. [Precision results](../experiments/openfold3_precision/README.md#collected-results-and-next-experiment).

Recommended work, in order:

1. Investigate the accepted QKV kernel's QK/softmax/PV scheduling using trained
   inputs and the existing exact reduction order. Compare against the accepted
   unroll4 implementation before running a complete model.
2. Attribute non-forward invocation time. The differences between per-run E2E
   and model timers are approximately **86.9-95.1 s** in the three optimized
   runs. These differences are not all removable startup overhead: loading,
   validation, features, serialization and supervision require separate timing.
3. Defer whole-PairFormer precision changes. Any narrower precision experiment
   must keep diffusion FP32, retain per-module/cycle coverage including chunk
   tuning, and report aligned/local/confidence deviations before an acceptance
   threshold or timing. Finite outputs alone do not qualify a mixed candidate.
4. Qualify maintained integration separately; promoting an already measured
   experimental composition is useful deployment work, not another speedup.

As a sensitivity example only, halving an operation occupying 35.37% of a trace
would reduce that trace's summed kernel time by 17.69%, or yield about 1.215x
at that boundary with everything else unchanged. It does not predict E2E gain;
the trace includes instrumentation and stalls, and the reduction is hypothetical.

For strict kernel work, keep FP32, recycles, diffusion steps, sample count and randomness fixed. Require
trained operator parity, full-model array/CIF parity and three alternating E2E
pairs before accepting a new speedup. The separate precision experiment explicitly
changes PairFormer arithmetic and needs its own quality policy. Use only HIP 7
under the latest user instruction and
verify the physical PCI mapping. Keep profiler runs separate from timed pairs.

## 2. Mini-ESM and Mini-default need current measurements

Mini-ESM N1000 has validated numerical integration, including fresh ESM2-3B
features, but no controlled paired E2E speedup. Its language-model work runs
during data preparation, outside `model_seconds`. A Rust N1000 C4/D1/S1 paired
baseline with separate ESM loading, embedding, folding and invocation stages
is the next useful investigation. Preserve ESM features and model semantics;
headroom remains unknown until measured.
[Mini-ESM timing and correctness scope](protenix-mini-esm.md).

Mini-default's historical N1000 medians are **27.949 s request / 15.391 s model**;
N1536 is **45.596 / 32.302 s**. Request timing and GPU-event model timing differ,
so subtraction alone is not a CPU startup profile. Remeasure the maintained
Rust entry first. Review later shared loading, layout and operator work against
Mini's actual dimensions; Base C10/D200 results cannot establish a benefit for
Mini C4/D1. Historical HIP/SMI monitor mapping also limits isolation evidence.
[Mini evidence](protenix-native-requests.md).

## 3. AtlasFold and Base have evidence of diminishing returns

**AtlasFold with AtlasLM-3B:** N1536 request time is **203.910 s**, compared with
553.359 s chunked native math (**2.714x**, sequential three-run sets). Final QKV
packing improved its own paired baseline by 1.0149x. Later normalization,
chunk-size and softmax-storage trials offered marginal/no benefit or extra
memory use. A packed LM-feature candidate changed trained projection outputs;
the exact alternative projected only about 0.070 s total saving, without a
full-model speedup measurement. Reopen this cycle only after a new stage profile
identifies substantial remaining work. All stochastic AtlasLM passes remain
required; the unchunked N1536 native route OOMs.
[Plateau audit](atlasfold-performance.md#atlaslm-follow-up-and-practical-plateau).

**Protenix Base:** maintained N1000 complete invocation is **110.025 s**, model
forward **86.894 s**. Sampled triangle BMM MFMA activity reaches 98.6-98.9%; CK
attention is about 75.1%. These particular BMMs have less obvious scheduling
headroom than OpenFold3's sampled fused QKV, although the observations are not
an architecture-wide efficiency comparison. Global diffusion attention has
lower utilization but only about 2.2% of summed trace kernel time.
[Hardware investigation](protenix-base-hardware.md).

Graph was 0.54% slower in three pairs, gate fusion regressed E2E, and transition
fusion delivered only 0.51%. The user closed the batching direction; its modest
throughput gain increased per-request latency substantially. Retain the qualified
route and reopen tuning only with new evidence. Preserve the explicit 0.1 Angstrom
aligned-RMSD/0.001 summary-confidence policy; native agreement is not bitwise.
[Qualification and rejected trials](protenix-base-optimization-audit.md).

## Models without a long-sequence performance claim

Tiny L124 validation does not establish long-sequence performance. OpenFold2
and ESMFold remain future adapters. Their integration and representative native
baselines must precede kernel ranking or published acceleration claims.
