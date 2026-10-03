# Checkpoint-covered construction experiment

## Finding

The current original-Torch comparison exposed approximately 18.6 seconds of
model construction before checkpoint loading. A subsequent instrumented
Mini-default construction attributes 18.013 of 18.713 CPU seconds to 621
truncated-normal initializer calls over 97,331,488 elements. The checkpoint
then covers every model state entry, with no missing or unexpected keys.

Installed SciPy 1.18.1 implements this sampler through `rv_continuous._rvs`:
one uniform draw per requested element followed by an inverse-CDF transform.
The private candidate retains those uniform draws and omits the distribution
transform and weight copy that the checkpoint will overwrite. Other
initializers, including Torch's ordinary Linear initialization, are unchanged.
This preserves random-state advancement rather than bypassing RNG consumption.

## Instrumented comparison

Base source `b54f653d24f1a84d842b6736a7f2a409dbf6b9da`; fresh processes,
warm filesystem caches, one CPU Torch thread, hipBLAS, FP32, TF32 disabled,
physical GPU 7 for model transfer, and continuous isolation checks on GPUs 6
and 7. Construction seed is 1031. Runner settings retain Mini-default v0.5.0,
C4/S1, one sample, request seed 101, and no MSA/templates, but no predictions
are performed. Timing includes cProfile instrumentation. These are individual
construction measurements, not a repeated-inference latency distribution.

The initial native profile is followed by candidate then native-repeat runs;
the last two use identical probe source.

| Stage | Native repeat | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| CPU construction | 18.693 s | 1.471 s | 12.708x |
| Construction and GPU transfer | 18.875 s | 1.650 s | 11.436x |
| Checkpoint load | 0.304 s | 0.301 s | 1.012x |
| Complete Runner creation | 19.190 s | 1.962 s | 9.783x |

Runner creation saves 17.228 seconds in this instrumented pair. Initializer
time falls from 17.999 to 0.678 seconds; both sides make 621 calls and account
for the same number of elements. Profiling identifies inverse-CDF-related
functions such as `_logsumexp`, `_norm_cdf` and `ppf_left` as native hotspots.

## Exact checks

- All 1,612 loaded state entries have the same combined shape/dtype/value hash.
- The model contains 134,067,007 parameter elements, including non-trainable
  Fourier embedding parameters.
- Python, NumPy and Torch CPU RNG digests match before and after construction.
- Checkpoint file hashes match; missing and unexpected key lists are empty.
- Both device isolation monitors pass in every run.

The model exposes no registered buffers, so the separate registered-buffer
digest is empty and is not additional numerical evidence. The probe does not
establish prediction agreement, GPU inference speed or request-level latency.

## Integration requirements

This remains a standalone experiment; runtime defaults are unchanged.

1. Limit the first implementation to the verified inference Runner/model and
   sampler contract. Unsupported versions or initializer arguments must use
   native construction.
2. Keep initializer interception scoped to construction, restore bindings on
   errors, and avoid changing unrelated training or concurrent construction.
3. Verify coverage of the checkpoint actually loaded before accepting the
   candidate. A non-strict checkpoint with missing keys must recover native
   initialization and preserve RNG state; ordinary Linear defaults cannot
   substitute for missing truncated-normal weights.
4. Record selected/fallback counts and separate construction, transfer, loading
   and complete request times. Keep kernels and model execution unchanged.
5. Validate incomplete checkpoints and failure cleanup, then compare full
   same-seed predictions and fresh-process latency at both long lengths.

The upstream inference loop resets the request seed before reading batches,
but this experiment additionally preserves constructor RNG advancement.
That loop behavior is not used as a substitute for the exact state checks.

```bash
HIP_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:Protenix python benchmarks/protenix/profile_construction.py \
  --output outputs/benchmarks/construction-native.json

HIP_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:Protenix python benchmarks/protenix/profile_construction.py \
  --skip-truncated --output outputs/benchmarks/construction-candidate.json
```

The probe rejects the skip experiment if checkpoint coverage is incomplete.
Compare the two reports' loaded-state and pre/post-RNG digests before treating
the experiment as successful. [Evidence](../provenance/protenix_construction_probe_b54f653.json)
retains all three profiles, source/checkpoint hashes, SciPy sampler source,
timings, coverage, state digests, and isolation records.
