# ESMC combined AMD SMI sampling

The native monitor now offers an explicit `snapshot-v1` operation. Each sample
opens one fresh AMD SMI session, enumerates devices, reads every BDF and PID list,
and emits a versioned envelope after successful shutdown. This replaces the
previous separate `list` and `process` sessions. It does not cache a mapping or
keep an SDK session alive. The reads are sequential, not an atomic hardware snapshot.

## Incremental E2E results

Bundled ESMC 6B, N1000 per input, B4 grouping, native BF16, seed101,
HIP6 / PCI `0000:8e:00.0` / SMI4. Each comparison uses three AB/BA/AB pairs,
fresh workers and query processes, unflushed caches, all original checkpoint
hashes and all exports. Offline compilation is excluded. Both arms use the same
Rust CLI and frozen model workers. Only monitor identity and protocol differ.

| Workload | Separate native queries | Combined query | Ratio |
| --- | ---: | ---: | ---: |
| One B4 group | 12.631937 s | 12.353837 s | **1.02251x** |
| 64 inputs, sixteen B4 groups, lazy FFN route | 22.472784 s | 22.235783 s | **1.01066x** |

Every pair is faster. B4 paired ratios are 1.05412x, 1.01934x and 1.01933x;
the first pair has a larger benefit than the other two. Resident64 paired ratios
are 1.01376x, 1.00493x and 1.02410x. Three pairs are limited evidence, not a
statistical significance claim. Do not multiply these ratios by historical results.

Per-sample query medians decrease from 350.933 ms (two previous native calls)
to 176.103 ms. Isolation sample medians increase **11 -> 15** for B4 and
**24 -> 34** for resident64. The scheduling interval stays 250 ms.
Model-forward medians are 1.292854 -> 1.273523 s and 9.258098 -> 9.226571 s,
respectively. Model operators and arithmetic are unchanged; the treatment targets
host query overhead, not GPU computation.

## Direct comparison with the original implementation

A separate three-pair comparison measures **24.235167 -> 12.250122 s E2E,
1.97836x** for original native B4 versus the full current composition.
Paired ratios are 1.92544x, 1.99216x and 1.86436x. The reference ranges from
22.715069 to 24.404236 s; the candidate ranges from 12.183830 to 12.586845 s.
The native reference retains its original loader, sequential imports, hash4,
inherited CPU affinity and original AMD SMI CLI. Both arms use the same final
Rust CLI, frozen workers, assets, input, grouping and seed.

The preceding 1.891x direct result was measured in another batch. Do not attribute
the entire ratio change to this treatment; the monitor-only comparisons above
isolate its contribution. No new direct-native resident64 ratio is claimed.
All 18 measured invocations / 432 prediction archives pass exact-array acceptance
and independent comparisons against official same-batch references.

## Protocol and qualification

Set `environment.FIREPROT_AMD_SMI_PROTOCOL` to `snapshot-v1` in the execution
profile. Missing or explicit `cli` keeps the previous two-query behavior.
Other values are rejected before inference; batch launch validates every profile
before starting its first job. No protocol is inferred from a filename, and a
failed combined query never falls back silently.

The helper supports:

```text
fireprot-smi-probe list --json
fireprot-smi-probe process --json
fireprot-smi-probe snapshot-v1 --json
```

A snapshot has exactly the root fields `schema_version: 1` and `devices`.
Each device contains `gpu`, `bdf` and `process_list` using the existing PID record
format. Rust validates all BDFs/indices and every device's process list, checks
that the selected physical device still has the same index, then removes only
the verified worker PID. Initial independent HIP verification still uses `list`.
Timeouts, process-group cleanup and output bounds remain unchanged.

Qualification covers 28 fake-SDK cases, five bad-argument cases, all-eight-device
mapping agreement, two real HIP6 PID creation/removal lifecycles, and real Rust
inference rejection of a controlled external HIP6 process before Start. The
helper retains the qualified AMD SMI 27.0.0 requirement. Rust tests exercise
wrong versions, malformed JSON, duplicate/missing/reordered devices, invalid PID
lists (including other devices), failed exit, timeout and absence of fallback.
CLI tests verify explicit selection, recorded protocol identity and early rejection.

Protocol changes require `monitor_treatment: true` and E2E scope in paired
comparisons, just like monitor executable replacement. Exact old/new values must
still be declared. This permission does not itself prove monitor equivalence.

## Entries and reproduction

Use the existing B4 request with `plm-esmc-fold2-6b-exact.json`. Use the existing
resident64-GEMM request with `plm-esmc-fold2-6b-resident-snapshot.json`.
The previous B4 profile is preserved as `plm-esmc-fold2-6b-native-monitor.json`;
`plm-esmc-fold2-6b-resident-monitor.json` also retains separate native queries.
All names above are under `configs/diamondhill`, except requests under
`configs/requests`. These absolute-path profiles target DiamondHill.

Example using the frozen, measured final CLI:

```bash
cd /media/IntelSSD/fireprot
cp configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json /tmp/esmc-snapshot.json
# Set a fresh run_id and output_dir in the copied request.
outputs/experiments/esmc-snapshot-monitor-20261016/fireprot-final \
  esm /tmp/esmc-snapshot.json configs/diamondhill/plm-esmc-fold2-6b-exact.json
```

The qualified executable is retained under
`outputs/experiments/esmc-snapshot-monitor-20261016`. Reproduction drivers under
`experiments/plm` build native/fake probes, qualify the SDK, prepare paired requests,
check real isolation, execute pairs and independently verify every exported byte
against official same-batch reference archives. The holder is the pinned HIP6
fixture from the preceding monitor experiment. Model weights are never rewritten.

The incremental experiment used `fireprot-bench`. The final `fireprot-final`
adds only comparison-policy protection for the new protocol variable; inference
and monitor code are identical. Frozen build/source hashes distinguish them.
The final policy replays both incremental reports without changes.
Measured scripts remain frozen remotely; the maintained preparation driver uses
the preserved prior profile so later default changes cannot move its baseline.

See [the prior native-monitor experiment](esmc-native-monitor.md) for the previous
implementation and [evidence](../provenance/esmc_snapshot_monitor_20261016.json)
for qualification, binary identities, exact acceptance and complete measurements.
