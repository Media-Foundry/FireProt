# OpenFold3 N1536 hardware diagnosis

The latest HIP-packing composition was profiled in two separate full-model runs
after the direct-native timing study. Both preserve all11 output arrays bitwise,
full CIF bytes and the qualified operator receipts. Workload: OpenBind-0 strict
deterministic FP32 with memory filling, N1536, R3/four trunk passes, D200/S1,
seed101, HIP7 on gfx90a MI250 (PCI0000:93:00.0, SMI5,104 CUs), unflushed caches.

## Trace and sampled coverage

The kernel trace contains807,382 dispatches. Summed kernel duration is248.550 s;
the interval union is247.825 s over287.021 s from first to last kernel (86.34%).
This includes initialization and instrumented stalls; it is neither model-only
utilization nor a replacement for uninstrumented E2E timing.

The counter run contains112 dispatches across49 kernel-name/grid combinations.
Every sampled dispatch has all nine requested metrics and a positive duration.
These combinations account for112.933 s, or45.44%, of trace kernel duration.
Occurrence10–12 filtering applies to kernel names, so important later GEMM
shapes are absent. Matching name/grid does not prove identical tensor strides
or execution phase. There is no model-wide counter average.

The counter profiler reported301 outstanding kernels and queue synchronization
timeouts during finalization. This limits completeness: the112 records above
are complete individual samples, not proof that every selected dispatch was
captured. The uninstrumented E2E study and exact model-output gates are separate.

## Measured priorities

| Sampled operator / grid | CU busy | Wave occupancy | MFMA utilization | HBM throughput |
| --- | ---: | ---: | ---: | ---: |
| Fused QKV /50,331,648 | 99.85% | 98.75% | 25.97% | 65.92 GB/s |
| Gate packing /3,145,728 | 98.87% | 34.04% | 0% | 390.48 GB/s |
| C128 HIP normalization /12,582,912 | 97.30% | 85.94% | 0% | 1347.55 GB/s |
| H16 bias packing /9,437,184 | 96.61% | 91.91% | 0% | 1226.59 GB/s |

Values are medians of three sampled dispatches. CU busy uses the installed SDK's
`SQ_BUSY_CU_CYCLES / GRBM_COUNT /104 *100` formula; there is no extra factor4.
Wave occupancy and MFMA utilization are separate metrics. Bandwidth derives from
`FetchSize + WriteSize` (KiB) divided by dispatch duration, in decimal GB/s.
No new peak-throughput calibration was performed.

Fused QKV is the largest named kernel:5,064 calls,87.922 s (35.37% of summed
trace duration), median15.529 ms. Its largest observed dispatch is2.360 s;
all stalls remain in the report. High CU activity together with low HBM traffic
and moderate MFMA utilization motivates examining arithmetic and synchronization
scheduling. These counters do not distinguish their contributions by themselves.

The source uses16 waves for QK/softmax but only two for PV. An eight-query,
512-thread fusion trial tested whether more independently schedulable blocks
offset extra redundant MFMA rows and QK barriers. It passed exact gates but
regressed complete attention latency. A second512-thread trial retained16 queries
and held two rows per wave; it also passed exact gates and regressed latency.
The existing1024-thread unroll4 remains selected. See the
[two measured rejections](../experiments/mfma_qkv/README.md#counter-guided-smaller-block-trials).

Adding batch is not supported as the first remedy for the large sampled QKV
grid: CU activity and wave occupancy are already high. This does not establish
that every OpenFold3 operator is compute-bound or rule out throughput benefits
for a separate multi-request workload.

## Follow-up on late GEMM occurrences

A separate full-model counter run selects occurrences1000–1100 of the two
largest GEMM names. It passes all11 arrays and full CIF bytes under the refreshed
complete environment inventory. However, the SDK again reports301 outstanding
kernels at finalization; only one of the two requested names has usable samples.

The main observed grid6,291,456 has93 complete samples: median5.738 ms,
CU busy99.73%, wave occupancy24.69%, MFMA utilization72.71%, HBM450.40 GB/s.
Eight additional samples belong to its smaller grid98,304. These observations
support investigating compute efficiency for that particular large GEMM; they
do not establish utilization for the missing GEMM family.
[Actual coverage and limitations](../provenance/openfold3_gemm_counters_20261010.json).

## Evidence and reproduction

- [Complete per-grid report and raw hashes](../provenance/openfold3_hardware_profile_n1536.json).
- [Profile collection](../experiments/openfold3_diffusion/hardware_profile.py).
- [Receipt validation and summarization](../experiments/openfold3_diffusion/summarize_hardware.py).
- Remote root: `outputs/experiments/openfold3-hardware-followup-20261010`.

The summary validates profiler/source receipts, raw CSV hashes and output pins;
it cross-checks dispatch counts and duration sums against SDK statistics. The
raw trace remains on DiamondHill. Instrumented timings must not enter an E2E
speedup claim.
