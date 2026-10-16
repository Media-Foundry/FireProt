# ESMC Rust host scheduling

The standalone ESMC B4 profile now uses eight Rust verification threads and a
worker CPU mask local to HIP6. Full original asset hashing, native BF16 model
semantics, output export and bitwise acceptance remain enabled. The qualified
HIP operators, pinned loader and import overlap are unchanged; no fixed-GEMM,
CK replacement or graph candidate is added.

## Complete invocation evidence

Each row is a separately scheduled three-pair AB/BA/AB comparison using four
N1000 sequences in the same B4 group, seed101, HIP6 / PCI `0000:8e:00.0`, fresh
workers and unflushed caches. All requested outputs and full original checkpoint
SHA-256 checks are included. There is no converted-checkpoint cache.

| Comparison | Baseline median | Candidate median | Speedup | Paired ratios |
| --- | ---: | ---: | ---: | --- |
| Python affinity pilot, four hash threads | 16.472502 s | 15.940837 s | 1.033352x | 1.0316x, 1.0373x, 1.0251x |
| Four -> eight Rust hash threads, inherited CPU mask | 16.299310 s | 12.935567 s | **1.260038x** | 1.3023x, 1.2027x, 1.2454x |
| Rust worker affinity, eight hash threads on both arms | 12.870973 s | 12.601278 s | **1.021402x** | 1.0262x, 1.0207x, 1.0129x |
| Original native -> final composition | 24.240289 s | 12.685461 s | **1.910872x** | 1.9081x, 1.9276x, 1.8513x |

Every pair improves. Independent collectors rehash **24 invocations / 96 output
archives**, compare every token/embedding byte to retained official same-batch
output, and check device identity, grouping and HIP operator coverage. Asset
provenance is identical across each pair; verification does not omit any file.
The direct-native comparison uses four hash threads and inherited CPU placement
for the reference, with the native loading path and import overlap disabled.

Do not multiply the incremental ratios or mix medians between rows. The directly
measured 1.911x supersedes 1.440x for standalone B4 only. Singleton, resident64,
ESMFold2 folding and other-model claims are unchanged. Earlier warm-forward
numbers remain separate operator measurements.

## What became faster

The hash-thread comparison leaves the supervisor median effectively unchanged
(9.257 -> 9.251 s). Median time outside that boundary falls from 6.869 to 3.684 s.
That remainder includes preflight and other launch setup, not just hashing.
The runtime still hashes every byte, checks before/after metadata, joins all
verification workers and rejects any failure before starting model execution.
The supported explicit range is now 1–8; the default API remains serial.

The initial affinity pilot measures checkpoint placement at 1.531 -> 1.239 s
median. It observes 295 threads with the full CPU mask in the baseline, versus
165 local-mask threads and two full-mask runtime threads in the candidate.
This is CPU scheduling evidence, not proof of a particular pinned-memory NUMA
allocation policy; no memory-binding policy is changed.

The final Rust affinity comparison has forward medians 1.295 -> 1.280 s and
request medians 3.739 -> 3.427 s. Direct-native forward medians are
1.456 -> 1.297 s. Most of the new E2E saving is host-side work, not GPU arithmetic.

## Rust ownership and validation

`FIREPROT_WORKER_CPU_AFFINITY` is an explicit profile environment setting. `inherit`
or absence preserves the original behavior; CPU IDs and ranges select a bounded
mask contained in the spawning thread's allowed set. Rust briefly applies it
around spawn, immediately restores the calling thread and independently checks
the main worker's actual mask before authorizing Start. The guard cannot move
between threads, and post-spawn restoration failures use normal child cleanup
and failed-run archival. The workspace still forbids unsafe code.

The promoted path does not use the pilot's Python affinity module. Nor does it
promise to keep every runtime-created thread inside the mask indefinitely.
See [launch provenance](execution-provenance.md#explicit-worker-cpu-affinity).

All 116 runtime/CLI tests pass, including serial/parallel provenance equality,
corrupting each asset under four/six/eight verification threads, malformed masks,
child inheritance, parent restoration, actual-mask mismatch and supervisor
integration. The 13 supervisor tests pass again after the final failure-archival
adjustment; final Clippy with `-D warnings` and formatting checks pass.
The isolated hash comparison uses the first build. Final affinity and native
qualification use the final build; each retains its own binary/source hashes.

## Entry point and fallback

The existing request remains valid:
`configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json`.
The public profile `configs/diamondhill/plm-esmc-fold2-6b-exact.json` sets:

```json
{
  "FIREPROT_PREFLIGHT_HASH_THREADS": "8",
  "FIREPROT_WORKER_CPU_AFFINITY": "64-127,192-255",
  "FIREPROT_ESMC_CPU_LOADING": "1",
  "FIREPROT_ESMC_IMPORT_OVERLAP": "1"
}
```

The CPU list is specific to DiamondHill. Use a new run ID and unused output
directory in a copied request. The reproducible command there is:

```bash
cd /media/IntelSSD/fireprot
outputs/experiments/esmc-host-20261016/native-final/qualified-hash8/target/release/fireprot \
  esm /tmp/esmc-request.json configs/diamondhill/plm-esmc-fold2-6b-exact.json
```

Use this measured CLI or rebuild the current Rust workspace; older CLIs reject
eight hash threads. The preceding frozen profile is retained as
`configs/diamondhill/plm-esmc-fold2-6b-startup.json`. With the new CLI, setting
four hash threads and `inherit` restores the preceding host settings.

Evidence: [sealed reports, builds and profile](../provenance/esmc_host_scheduling_20261016.json).
Remote roots are `outputs/experiments/esmc-affinity-20261016` for the initial pilot
and `outputs/experiments/esmc-host-20261016` for Rust qualification. Early unused
staging directories and both build identities remain retained. Network timeouts
affected observation only; completed GPU jobs were not restarted.
Maintained reproduction drivers now read their corresponding historical profile
generation instead of the moving public profile. Original measured scripts
remain sealed in each experiment, including `prepare-measured.py` copies.
