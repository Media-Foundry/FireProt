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

The initial experiment was standalone. The subsequent opt-in runtime
integration below implements these requirements; runtime defaults are unchanged.

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

## Opt-in runtime integration

`FIREPROT_ENABLE_FAST_MODEL_INIT=1` enables the construction adapter for
Mini-default v0.5.0 FP32. The default remains disabled. The adapter verifies
all model Python files, the Runner construction/loading methods, initializer
source, SciPy version and sampler implementation before installing hooks.
Unknown contracts use native construction. The optimization is independent
of GPU kernels and does not reduce recycling or diffusion steps.

Interception is scoped to the construction call. Eligibility requires the
main Python thread with no other active Python threads and the global NumPy
sampler. Unsupported initializer arguments use native initialization.
Checkpoint coverage is observed on the actual loaded state dictionary. For
non-strict missing keys, the entire native model is reconstructed with the
original RNG state and the already deserialized checkpoint is reapplied;
the caller's RNG states are restored after recovery. Strict loading errors
propagate after temporary bindings are cleaned up.

The runtime manifest records source contracts, attempts, selected/recovered
constructions, native fallback reasons, skipped element counts, construction
including transfer, and checkpoint loading including recovery. CPU tests
cover complete/incomplete checkpoints, RNG equality, failure cleanup,
unsupported configurations, additional Python threads and ordinary tensors.

## Fresh-worker request comparison

The integrated candidate was measured against the same FireProt source with
the construction flag disabled. Base commit is `3971121`; the exact uncommitted
adapter and runtime hashes are recorded in the evidence, so the base commit
alone does not identify the candidate. Existing CK/HIP inference paths are
identical on both sides.

Mini-default v0.5.0, FP32, TF32 off, hipBLAS, C4/S1, one sample, seed 101,
no MSA/templates, zero warmups and one prediction per fresh worker. Filesystem
and compiled-kernel caches are warm. GPU 7 executes serially with continuous
isolation monitoring of GPUs 7 and 6; all checks pass. N1536 runs enabled then
disabled; N1000 uses the reverse order.

| Length | Flag disabled | Flag enabled | Request speedup | Saved |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 47.326 s | 30.435 s | 1.555x | 16.891 s |
| 1536 | 70.328 s | 52.097 s | 1.350x | 18.231 s |

These are single paired observations, not medians or a latency distribution.
The timing is the harness worker's `process_wall_ms`: setup, model loading,
feature preparation, one prediction and output writing. It excludes parent
launch/preflight time and is not a cold-filesystem or cold-compilation result.
This is incremental improvement over current FireProt, not a new comparison
against original Torch. It does not establish faster warm-model execution.

Both enabled runs select all 621 eligible initializers (97,331,488 elements),
have zero missing checkpoint keys, and take about 1.48 s for construction
including transfer plus 0.30 s for loading. Both output comparisons pass the
unchanged 0.001 Angstrom aligned-RMSD and 0.001 confidence gates. This verifies
agreement with the disabled path on these performance inputs, not experimental
structure accuracy. Keep the flag opt-in pending broader model/version coverage
and repeated fresh-worker measurements.

[Integration evidence](../provenance/protenix_fast_init_3971121.json) includes
the launch script, source hashes, complete configuration, per-run counters,
timing, device isolation records and output comparisons.
