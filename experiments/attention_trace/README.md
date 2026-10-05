# Bounded OpenFold 3 attention observation

Status: three CPU observer tests pass. The N1536/HIP7 diagnostic is launched
after successful completion of the conditioning-cache paired benchmark, in
`/tmp/fireprot-attention-trace-v1`. No accepted GPU trace, complete-model parity,
kernel correlation or speedup is established yet. Source/request/archive pins
are in `provenance/openfold3_followup_diagnostics_preparation.json`.

`probe.py` wraps the existing `AttentionDispatcher` within its existing
source-pinned `attention_scope`. It delegates the original operations unchanged,
labels QK/PV einsums, records ATen GEMM/packing tensor shapes and strides, and
exports a raw Chrome trace for CPU/GPU correlation. It retains no tensor values.
The small Python boundary is required to observe upstream Torch calls; Rust
continues to own process launch, isolation, artifact sealing and acceptance.

## Diagnostic deployment after the timing sequence

1. Create a fresh immutable copy of the accepted combined HIP/cache worker.
   Copy `probe.py` into that copy's `python/fireprot_workers/attention_trace.py`.
   Do not modify the production or live benchmark worker.
2. In the copied `forward_with_backend`, retain `dispatcher` for its normal
   manifest, but pass `observed` into `attention_scope`:

   ```python
   from .attention_trace import AttentionTrace
   observed = AttentionTrace(dispatcher, output, (32, 4, 1536, 32))
   with attention_scope(observed):
       result = forward(model, *args, **kwargs)
   observed.require_complete()
   return result
   ```

   Keep the existing `try/finally` that writes the operator manifest. Retain
   conditioning reuse and all strict model settings. Add `attention-trace.json`
   and `attention-trace.chrome.json` to the copied transport's sealed diagnostics.
3. Pin the new source tree/profile and use a fresh run ID under Rust supervision.
   Require all 11 arrays to equal the strict native reference, and verify the
   expected cache/HIP counters. The instrumented run is excluded from E2E ratios.
4. Read `attention-trace.json` before assigning any kernel name to QK or PV.
   Inspect the raw trace's ancestry and correlation IDs. Duplicate or missing
   ROCTracer flows invalidate the affected association; aggregate GEMM names
   alone do not establish a mapping.

The first sample selects the conservative 32-row chunk already observed by
the N1536 dispatcher. This call can belong to autotuning. `occurrence` selects a
later matching call explicitly but does not establish that tuning has finished.
Do not extrapolate its layout or kernel choice to all other chunk sizes. A
subsequent probe should target actual steady-state chunks if they differ.
Missing calls, incomplete observations and exceptions must not be accepted as
successful diagnostic evidence. Failed calls consume the one allowed attempt.

Profiler synchronization, shape retention and mode dispatch perturb latency and
memory. This bounded observation therefore cannot supply an E2E speedup claim
or whole-model hotspot ranking. It supplies the missing layout and correlation
evidence needed before an isolated HIP/CK-style GEMM experiment.

## CPU verification

```bash
python -m unittest experiments.attention_trace.test_probe -v
```

The tests use head-interleaved inputs and check exact results, unchanged RNG,
QK/PV labels with observed BMMs, one selected attempt, exception recording and
unmatched-call pass-through. They do not validate ROCm profiler behavior.
