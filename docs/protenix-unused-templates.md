# Omission of unused Mini template features

## Eligibility and implementation

`FIREPROT_SKIP_UNUSED_TEMPLATES=1` enables an experimental data-adapter path.
It accepts only `protenix_mini_default_v0.5.0`, FP32, configured and actual
`use_template=False`, and `model.template_embedder.n_blocks=0`. The consumer's
`TemplateEmbedder.forward` source SHA-256 must match the verified early-return
implementation; unknown source or wrapped methods use native preparation.
The flag remains opt-in after the full-request validation below; coverage is
limited to the two tested inputs and this verified consumer.

Independent Protenix patch 0007 exposes this policy only when the FireProt
triangle backend is requested. The adapter returns an empty template result;
the native dataset then creates its usual small dummy template fields. Other
features follow the original preparation path. Template-enabled configurations,
Base/other model names, non-FP32 configurations and incomplete settings retain
the native builder. No shared template method is temporarily patched.

The model constructs `TemplateEmbedder` from this configuration and returns
zero before reading any template tensor when `n_blocks=0`. This consumer
invariant justifies omitting its otherwise unused fields; it does not justify
omission for template-capable models. Runtime manifests report per-process
native/skipped calls and the consumer fingerprint. Current model benchmarks
use `num_workers=0`; worker-local counts are not an aggregate across processes.

## CPU feature validation

Base `74697bc`, independent Protenix `932de05`, input hashes in the evidence.
Single-thread CPU, native feature dtypes, seed 101 reset per call, no MSA/RNA
MSA/template/ESM search, one unmeasured native warmup per length, then native,
skip, native order. Process/filesystem caches are warm. No checkpoint, recycle,
diffusion step or prediction sample applies. The GPU is never initialized.
These timings are individual uninstrumented calls, not latency distributions.

| N | Native before / after | Skip | Native / skip logical feature bytes |
| --- | ---: | ---: | ---: |
| 1000 | 2.854 / 2.862 s | 1.335 s | 1,280,572,192 / 577,692,192 |
| 1536 | 6.381 / 6.352 s | 2.607 s | 2,975,649,984 / 1,316,425,920 |

All non-template features (including nested fields and metadata), atom
coordinates/annotations/bonds, and Python/NumPy/Torch CPU RNG states have
identical hashes in each three-call comparison. Template fields are explicitly
excluded from the retained-feature digest because their schema changes to the
native dummy schema; both old and new field inventories are recorded. The
verified consumer returns zero for both forms. No inference speedup or
prediction agreement is established by this CPU experiment alone.

Ten unit tests cover the skip, all eligibility failures, consumer mismatch,
exact native return values and exception propagation. The full local suite
passes 155 tests with 53 HIP-gated skips.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_unused_templates.py \
  --output outputs/benchmarks/unused-templates.json
```

[CPU evidence](../provenance/protenix_unused_templates_cpu_74697bc.json) retains
source/input hashes, per-call native stage timers, complete tensor inventories,
retained-feature/atom/RNG hashes and dispatch counters.


## Fresh-worker single-request validation

Candidate `40e535a`, independent Protenix `932de05`. Each length uses three
pairs of fresh workers, one prediction per worker, zero warmups, alternating
A/B order. Both sides use Mini-default v0.5.0, FP32, TF32 off, hipBLAS, C4/S1,
one sample, seed 101, no MSA/RNA MSA/templates/ESM, and the current CK/HIP paths.
Fast model construction, pair-layout views, transposed LayerNorm and its
subgroup kernel are enabled on both sides. Only
`FIREPROT_SKIP_UNUSED_TEMPLATES` changes. Filesystem and compiled-kernel caches
are warm; model, CCD and allocator state are fresh per worker. GPU jobs run
serially on physical GPU 7 with GPU 6 monitored. All isolation checks pass.

| N | Native preparation request median | Skip request median | Request speedup | Disabled / enabled ranges |
| --- | ---: | ---: | ---: | --- |
| 1000 | 30.317 s | 28.637 s | **1.059x** | 30.004–30.357 / 28.614–28.766 s |
| 1536 | 51.565 s | 47.721 s | **1.081x** | 51.472–51.720 / 47.424–47.800 s |

These are `process_wall_ms` worker timings, including setup, model loading,
feature preparation, prediction and output writing. They exclude parent
launcher/preflight time and are not service/network latency. The narrower
`inference_wall_ms` medians are 26.964 → 25.284 s and 48.225 → 44.364 s.
First-forward GPU-event medians are 15.872 → 15.722 s and 33.605 → 33.349 s;
they include first-use overhead and must not be compared directly with the
previous warm-model medians. This change primarily improves request setup.
It does not reduce recycle count or change a GPU arithmetic kernel.

Model-call peak allocated memory changes from 8.570 → 8.457 GB at N1000 and
19.382 → 19.056 GB at N1536 (decimal GB). These are model-call peaks, not CPU
feature bytes or process-wide GPU peaks. The upstream model deletes template
features after the trunk, so the peak reduction need not equal the feature
size reduction.

All six prediction comparisons pass the locked 1e-3 Å aligned-RMSD and 1e-3
confidence gates. Maximum aligned RMSD is 0.00009213 Å at N1000 and
0.00014372 Å at N1536; maximum confidence delta is 7.63e-6 and 3.82e-5.
Every enabled worker reports exactly one skipped builder, every disabled
worker exactly one native builder, and every worker selects fast construction
once. These are numerical regression checks on development performance cases,
not independent experimental-coordinate accuracy validation.

[Full-request evidence](../provenance/protenix_unused_templates_model_40e535a.json)
contains all twelve manifests, six output comparisons, source/input hashes,
per-call timing and memory, dispatch counters, isolation reports, and the
exact serial reproduction script. The feature-only evidence above remains a
separate measurement. These incremental ratios must not be multiplied by
historical original-Torch ratios to claim a new controlled comparison.

Enable with Protenix patch 0007 and the FireProt backend:

```bash
export FIREPROT_SKIP_UNUSED_TEMPLATES=1
```

Set the flag to `0` or unset it to restore native template preparation.
