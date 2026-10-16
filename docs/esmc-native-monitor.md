# ESMC native AMD SMI monitoring

The qualified ESMC profiles replace repeated Python AMD SMI CLI startup with a
small native query executable. Rust still owns physical-device matching,
process isolation, sampling, timeouts and output acceptance. Model execution,
precision and operators do not change in the incremental comparison.

## Measured E2E

Conditions: bundled ESMC 6B, N1000 per input, B4 grouping, native BF16, seed101,
HIP6 / PCI `0000:8e:00.0` / SMI4. Each row is a separate three-pair AB/BA/AB
comparison with fresh workers/query processes, unflushed caches, full original
checkpoint hashes and all exports. Offline builds are excluded; library loading
and queries are included.

| Comparison | Reference E2E | Candidate E2E | Ratio |
| --- | ---: | ---: | ---: |
| Current B4, CLI -> native monitor | 12.695923 s | 12.432903 s | **1.02116x incremental** |
| Current resident64 FFN, CLI -> native monitor | 22.793357 s | 22.232328 s | **1.02523x incremental** |
| Original native B4 -> full current composition | 23.913582 s | 12.646611 s | **1.89091x direct** |

All incremental pairs are faster: B4 ratios are 1.02054x, 1.01669x, 1.02116x;
resident64 ratios are 1.02941x, 1.02411x, 1.00596x. The last resident candidate
is slower than its preceding calls but still faster than its paired control.
Three pairs do not establish distributional significance.

Direct-native paired ratios are 1.92362x, 1.89091x and 1.87953x. The historical
1.911x result used medians 24.240289 -> 12.685461 s in a different measurement
batch. Both reference and candidate timings changed. Report the current direct
**1.891x**, without multiplying incremental ratios or mixing batches' medians.
These observations do not establish the cause of cross-batch timing variation.

Monitor-only model-forward medians are 1.290276 -> 1.303267 s for B4 and
9.252227 -> 9.222229 s for resident64. This is a host E2E improvement, not a
new GPU-kernel speedup. The direct-native forward boundary is
1.454513 -> 1.331630 s; previous warm-operator measurements remain separate.

All 18 measured invocations / 432 prediction archives pass Rust exact-array
acceptance. Independent collectors rehash them and official same-batch references,
checking every token/embedding byte, dtype, grouping, asset provenance, physical
GPU and operator coverage. No tolerance, checkpoint, output or layer is removed.

## Contract and qualification

`kernels/amd/tools/fireprot_smi_probe.cpp` supports only `list --json` and
`process --json`: the subset consumed by Rust, not the entire AMD SMI CLI.
It emits GPU indices/BDFs or PID membership, never utilization counters.
Enumeration follows the installed binding's order. Only the query process's
own PID is excluded, matching the binding; Rust separately handles the worker PID.

The probe requires AMD SMI **27.0.0**. Process buffers grow on the documented
out-of-resources status, with bounded size/retries. Duplicate/null devices,
invalid BDFs/counts/PIDs, unknown versions/statuses, continuing growth and shutdown
errors fail without partial successful JSON. Rust's 250 ms scheduling interval,
timeouts, output limits and per-sample PCI-mapping checks remain unchanged.

Qualification includes:

- 28 CPU API cases: three successful cases and 25 rejection cases; five additional
  invalid-argument cases. The fake library is never linked into the real probe.
- Original/native mapping agreement for all eight indices/BDFs, plus empty HIP6 PID sets.
- Two real HIP6 process lifecycles. An immediate new native query detects the PID
  in about 0.185 s and agrees with alternating CLI observations; both detect removal
  after exit. This tests PID membership, not utilization-counter timing guarantees.
- A real Rust inference conflict test: an extra controlled HIP6 PID is rejected
  before Start authorization, with no predictions or accepted receipt. Cleanup
  returns the device to idle.
- Five Rust comparison tests, 15 CLI integration tests, Clippy with `-D warnings`,
  formatting and request validation.

Idle query medians improve 0.302333 -> 0.175614 s for `list` and
0.299486 -> 0.175978 s for `process`. Queries often overlap model execution, so
their savings cannot simply be added to E2E. Actual isolation sampling increases
from 8 to 11 per B4 invocation and from median 18 to 24 for resident64.

## Explicit comparison policy

Default comparison APIs still protect the monitor executable. The optional
`monitor_treatment: true` is accepted only for E2E. Both arms must already have
an `amd-smi` input identity, and exact old/new identities must be declared as
`launch.inputs.amd-smi`. HIP probe, supervisor and worker-entry identities remain
protected. Baseline-repeat checks keep default protection. The opt-in and
identities are archived. This permission does not establish monitor equivalence;
the independent qualification above does.

## Entry and fallback

The existing B4 request remains valid. The preserved
`configs/diamondhill/plm-esmc-fold2-6b-native-monitor.json` uses this measured native
probe. The current default has advanced to the [combined snapshot protocol](esmc-snapshot-monitor.md). Copy the request and select a fresh run ID/output directory:

```bash
cd /media/IntelSSD/fireprot
cp configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json /tmp/esmc-b4.json
# Edit run_id and output_dir in the copied request.
outputs/experiments/esmc-native-monitor-20261016/fireprot-bench \
  esm /tmp/esmc-b4.json configs/diamondhill/plm-esmc-fold2-6b-native-monitor.json
```

For resident64, pair `configs/requests/plm-esmc-fold2-6b-resident64-gemm.json`
with `configs/diamondhill/plm-esmc-fold2-6b-resident-monitor.json`.
The preceding B4 profile is preserved byte-for-byte as
`plm-esmc-fold2-6b-host-scheduling.json`; the resident-GEMM profile also retains
the original CLI. Other model profiles are unchanged.

These are DiamondHill deployment examples. The native probe is an ELF executable
with the build service's historical `.so` artifact suffix. Its hash is part of
Rust launch identity. The AMD SMI library hash is recorded and checked before/after
experiments, but this is not a complete dynamic-library or driver lock.

## Reproduction and retained failures

Remote root: `outputs/experiments/esmc-native-monitor-20261016`.
Native/fake builds are under `build-v2`; the holder's final receipt is
`hold-build-receipt-v3.json`. The first build rejected a deprecated linker-path
flag under `-Werror`. The initial holder lacked a HIP runtime rpath and failed
before allocation; only that holder was rebuilt. The real probe bytes are unchanged
across final qualification and E2E. Maintained probe source is byte-identical to
the frozen `esmc_smi_probe.cpp`.

Build/test/pair/collector drivers live under `experiments/plm`; their measured
versions remain frozen remotely. Historical drivers now use the preserved
host-scheduling profile. Rust source/binary identity, qualification, isolation
failure and measurements are retained in the [evidence](../provenance/esmc_native_monitor_20261016.json).
