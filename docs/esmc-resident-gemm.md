# ESMC resident-batch FFN composition

This experiment adds the previously exact lazy hipBLASLt FFN up/gate projection
to the [residual/SwiGLU pointwise route](esmc-pointwise.md). It improves the
measured resident64 workload slightly. At this experiment's completion the default
single-group profile and its **1.911x direct-native E2E** result were unchanged.
The subsequent [native-monitor update](esmc-native-monitor.md) is measured separately.

## Complete invocation

Both arms retain residual vector4, SwiGLU vector8, the pinned loader, import
overlap, eight full-file verification threads and Rust CPU placement. Only
`FIREPROT_ESMC_FIXED_GEMM` and the declared profile hash differ. The candidate
uses the original pinned library and algorithm 4139; it creates its plan at the
first FFN projection after native QKV and reuses it across subsequent groups.
No CK replacement, online tuning or precision change is involved.

Conditions: bundled ESMC 6B, N1000, B4 grouping, native BF16, seed101, HIP6 / PCI
`0000:8e:00.0`. Three AB/BA/AB pairs per workload use fresh workers and unflushed
caches. Full original checkpoint hashes, normal startup, shared-library and
plan loading, model forward and every requested output export are included.
Offline library compilation is excluded.

| Workload | Pointwise route E2E | With lazy FFN E2E | Incremental ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| Four inputs, one B4 group | 12.736230 s | 12.648520 s | 1.00693x | Mixed; do not change default |
| 64 inputs, sixteen B4 groups | 23.168709 s | 22.914643 s | 1.01109x | Separate experimental opt-in |

The B4 paired ratios are 1.00806x, 0.99994x and 1.00693x. Its second candidate
is 0.705 ms slower, so this is not a consistent single-group improvement.
The resident64 paired ratios are 1.03647x, 1.01109x and 1.00612x; all three are
faster. Candidate maximum is 22.958893 s versus baseline maximum 23.796241 s.
The baseline's first call is noticeably slower than its next two; retain the
complete observations rather than presenting the largest ratio as typical.

Model-forward medians improve 1.247357 -> 1.219238 s for B4 (1.0231x), and
9.789302 -> 9.283599 s for resident64 (1.0545x). These are forward boundaries
within complete invocations, not separate warm-forward microbenchmarks.
Do not multiply this experiment's ratios by preceding pointwise or native ratios.
No new direct-native resident64 performance comparison is claimed.

## Correctness and plan lifetime

All 12 measured invocations / 408 prediction archives pass Rust exact-array
acceptance. The independent collector rehashes these archives and 68 official
same-batch native reference archives, comparing every token and embedding byte.
All 476 archives match, including original dtypes and signed-zero representation.
Input identities/grouping, identical asset provenance, GPU PCI identity and
operator coverage are also verified.

Each group records 80 RoPE calls, 80 vector SwiGLU calls, 160 fused residual
updates and, for the candidate, 80 direct FFN projections. The plan is created
only in the first group and reused for all remaining groups. The qualified
algorithm reports zero workspace; the native support check and version/revision
guards still run. The collector explicitly verifies this zero-workspace case.
Plan initialization takes approximately 2 ms, rather than repeating enumeration
for each group. See [earlier initialization diagnosis](esmc-gemm-composition.md).

## Experimental entry

The new profile preserves the measured candidate profile bytes. It is a
DiamondHill example using pinned experiment assets and the frozen deployment.
Copy the request and choose a fresh run ID and output directory:

```bash
cd /media/IntelSSD/fireprot
cp configs/requests/plm-esmc-fold2-6b-resident64-gemm.json /tmp/esmc-resident64-gemm.json
# Edit run_id and output_dir in the copied request.
outputs/experiments/esmc-resident-gemm-20261016/qualified-resident-gemm/target/release/fireprot \
  esm /tmp/esmc-resident64-gemm.json configs/diamondhill/plm-esmc-fold2-6b-resident-gemm.json
```

Use the frozen CLI and worker together. Set `FIREPROT_ESMC_FIXED_GEMM=0` to
select the measured pointwise control in that deployment. The preceding
pointwise request/profile remains available for reproducing its own results.

## Remaining host cost: monitor diagnostic

An additional instrumented B4 call uses the unchanged default exact route and
wraps the original `/usr/bin/amd-smi` executable to record subprocess timing.
Arguments, stdout, stderr and exit status are forwarded unchanged. Rust retains
every device-mapping and process-isolation check. Its four outputs match the
official same-batch native reference bytewise.

The diagnostic performs seven isolation samples:

| Operation | Calls | Median subprocess time | Cumulative time |
| --- | ---: | ---: | ---: |
| `list --json` | 8 | 0.311816 s | 2.562860 s |
| `process --json` | 7 | 0.319324 s | 2.249630 s |

The combined 4.812490 s is **not recoverable E2E time**: these synchronous host
queries often overlap worker execution. The wrapper also adds overhead, so its
12.825706 s invocation is a diagnostic, not a new performance comparison.
It identifies subprocess-based monitoring as a candidate for further profiling.
No monitor checks or sampling frequencies have been reduced, and no alternative
monitor implementation has been qualified by this experiment.

## Evidence and reproduction

[Sealed reports and source identities](../provenance/esmc_resident_gemm_20261016.json)
cover the deployment, paired execution, independent collector, plan reuse and
monitor diagnostic. Remote root:
`/media/IntelSSD/fireprot/outputs/experiments/esmc-resident-gemm-20261016`.

`prepare_esmc_resident_gemm_pairs.py` creates a fresh isolated deployment and runs
both workloads sequentially. `collect_esmc_resident_gemm_pairs.py` rechecks the
archives; its frozen remote filename is `collect_esmc_resident_gemm_pairs_final.py`.
The earlier, unused collector incorrectly assumed workspace must be nonzero;
the native support-check contract and actual algorithm require zero here.
The correction precedes collector execution and does not change any inference.
`profile_esmc_monitor.py` creates a separate diagnostic output directory and
retains the wrapper source plus every timed query record.
