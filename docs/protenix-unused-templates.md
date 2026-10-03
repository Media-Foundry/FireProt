# Omission of unused Mini template features

## Eligibility and implementation

`FIREPROT_SKIP_UNUSED_TEMPLATES=1` enables an experimental data-adapter path.
It accepts only `protenix_mini_default_v0.5.0`, FP32, configured and actual
`use_template=False`, and `model.template_embedder.n_blocks=0`. The consumer's
`TemplateEmbedder.forward` source SHA-256 must match the verified early-return
implementation; unknown source or wrapped methods use native preparation.
The flag defaults off pending full-request validation.

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
