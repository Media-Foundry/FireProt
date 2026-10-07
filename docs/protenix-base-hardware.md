# Protenix Base hardware utilization and batching investigation

Status: investigation complete. The user closed this direction on 2026-10-07;
no batching change is promoted into the maintained entry.

## Measured configuration

Maintained Base-default v0.5.0, umem N1000/7979 atoms, FP32 C10/D200/S1,
seed 101, strict determinism and memory filling, HIP7 on DiamondHill. HIP7 maps
to PCI `0000:93:00.0`, AMD SMI device 5, with 104 compute units and 64 GiB-class
memory. The qualified CK/HIP/Lt composition and template omission are enabled;
Graph is off. No other FireProt GPU experiment overlaps these runs.

Two separate diagnostic runs collect a complete kernel trace and selected
hardware counters using the installed rocprofv3 1.3.5 SDK. Both reproduce all
nine arrays of the maintained optimized reference bitwise. The profiler is
injected through explicit profile environment settings so the real worker PID,
Rust supervision and device identity checks remain intact. Production profiles
are unchanged. Instrumented durations are not E2E performance measurements.

## Kernel-level measurements

Values below are medians of three sampled dispatches of each listed kernel/grid
shape, at kernel occurrences 10–12. CU/SIMD active is the installed SDK's
`SIMD_UTILIZATION` expression times 100. Wave occupancy measures resident waves
against the architectural maximum of 32 waves per CU; it is a different metric.
MFMA active measures matrix-pipeline busy cycles. Bandwidth is derived from
`(FetchSize + WriteSize) * 1024 / duration`, using counter sizes in KiB.

| Kernel | CU/SIMD active | Wave occupancy | MFMA active | HBM traffic rate |
| --- | ---: | ---: | ---: | ---: |
| CK triangle attention | 99.7% | 49.2% | 75.1% | 482 GB/s |
| Triangle BMM, Lt outgoing/incoming variants | 99.4–99.5% | 24.3–36.3% | 98.6–98.9% | 290–302 GB/s |
| Dual projection | 99.0% | 46.6% | 70.6% | 415 GB/s |
| QKV projection | 99.3% | 93.3% | 73.0% | 720 GB/s |
| Pair transition projection | 99.8% | 47.4% | 68.0% | 408 GB/s |
| Pair LayerNorm | 98.8–98.9% | 90.4–90.8% | 0% | 1,398–1,403 GB/s |
| Triangle residual/gate | 99.5% | 78.3% | 0% | 1,216 GB/s |
| Global diffusion attention (`attn_fwd`) | 61.5% | 7.7% | 19.5% | 206 GB/s |

An independent, uninstrumented calibration on the same GCD reaches approximately
**1,250 GB/s** for a 1 GiB streaming copy (read plus write traffic) and **36.88
TFLOP/s** for an 8192-square FP32 hipBLAS GEMM. These are empirical reference
points, not theoretical hardware ceilings. Other access patterns can exceed the
copy reference; the measured LayerNorm bandwidth is not an impossible percentage.

The padded triangle contractions perform approximately 256 GFLOP in 7.45–7.51 ms,
about 34 TFLOP/s, consistent with their very high MFMA utilization. The bandwidth
figures and MFMA metrics strongly support a compute-bound classification for
these BMMs. CK attention and projection kernels also keep the CUs active and use
matrix execution substantially, while staying below the streaming HBM rate.
This argues against HBM bandwidth saturation as their main limitation, but does
not isolate every LDS/cache/latency or instruction-scheduling stall.

LayerNorm and residual work are bandwidth-oriented. Global diffusion attention
has a much smaller grid and low occupancy/utilization, leaving more scope for
concurrency or a better kernel. It accounts for about 2.2% of summed kernel time
in this trace, so it does not explain the majority of complete inference time.

**Conclusion:** this workload has mixed bottlenecks. The large recycled kernels
already occupy the GCD and are mainly limited by matrix/other on-chip execution;
several memory-streaming kernels are bandwidth-limited. Low wave occupancy alone
does not mean half of the CUs are idle: the BMMs reach approximately 99% MFMA busy
with only 24–36% resident-wave occupancy.

## Whole-trace scope and limitations

The trace contains 537,608 dispatches. Kernel durations sum to 84.419 s; their
union is 84.416 s within a 106.231 s first-to-last-kernel span, about 79.5%.
That span includes preparation and host gaps, and instrumentation changes
execution. It is neither uninstrumented model utilization nor proof that batching
can recover the entire remaining 20.5%. The largest individual kernel family is
CK triangle attention at 24.5% of summed kernel time. Deterministic fill kernels
also account for 144,004 calls and about 6.2% of summed kernel time; memory filling
remains enabled in all compared scientific settings.

The counter profiler reported outstanding-queue warnings during finalization.
The process nevertheless exited successfully and its predictions passed exact
validation. The report uses 88 sampled dispatch records, each with all nine
requested metrics and positive durations; each hotspot in the table has three
samples. This is sampled hardware evidence, not a complete counter timeline.

[Hardware evidence](../provenance/protenix_base_hardware_profile.json) retains
metric definitions, source artifact hashes, raw-file locations, per-grid samples,
resource usage, calibration and the profiler warnings. AMD's
[counter reference](https://rocm.docs.amd.com/en/docs-10.1.0/reference/gpu-arch/mi300-mi200-performance-counters.html)
and [kernel filtering guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.1/how-to/kernel-naming-filtering.html)
describe the counters and sampling controls. The installed SDK expression is
used for CU/SIMD active; no extra factor of four is applied to it.

## Batch experiment boundary

The current maintained entry accepts one sequence-only request and enforces
exclusive GPU supervision. Increasing a model's diffusion sample count changes
the scientific workload and is not input batching. True tensor batching is not
implemented by this experiment.

The isolated Rust harness `experiments/protenix_base/batch_throughput.rs`
compares two serial fresh-process requests with two concurrent fresh-process
requests on the **same HIP7 GCD**, in serial/concurrent/concurrent/serial group
order. Every member invokes the unchanged maintained prediction function with
identical scientific settings. Rust owns process launch, timing, explicit cohort
monitoring and output checks; unrelated GPU PIDs fail the trial. The experiment
does not weaken the maintained supervisor or emit maintained acceptance receipts.

Timing includes Python startup, loading, feature preparation, prediction and
serialization, but excludes maintained Rust preflight. Therefore its throughput
ratio must not be substituted for the published 3.472x E2E comparison. Every
member must reproduce all nine arrays and the complete summary-confidence object
exactly. The completed results are recorded below.

### Completed throughput comparison and decision

All four groups complete successfully, covering eight requests with exact arrays,
exact summary confidence, and independently verified CIF/NPZ atom binding. Two
serial/concurrent comparisons in opposite order give throughput ratios of
**1.08825x** and **1.10101x**.

| Metric | Serial requests | Two concurrent requests on HIP7 |
| --- | ---: | ---: |
| Wall time for two requests, median | 211.679 s | 193.385 s |
| Per-request latency, median | 105.722 s | 192.542 s |
| Per-request model forward, median | 86.797 s | 172.857 s |

Overall throughput improves **9.46%**, while per-request latency increases by
about 82%. Doubling the solo model time nearly equals the concurrent model time;
this is consistent with overlapping loading/host work rather than a material
increase in GPU model throughput. It does not establish gains from true tensor
batching or persistent model reuse.

**Decision: retain the maintained single-request route and close this direction.**
The experiment remains available as evidence; the user requested moving on to
OpenFold optimization. The published Base 3.472x complete-invocation speedup is
unchanged. [Batch evidence](../provenance/protenix_base_batch_throughput.json).
