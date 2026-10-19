# FireProt gfx11 handoff

Snapshot: 2026-10-19 UTC. Latest copy experiment base revision: `23133ea` (`master`).
The checkpoint, softmax and reuse qualifications are on GitHub.
This document also includes the 2026-10-19 copy follow-up. The original 2026-10-16
snapshot is retained at
`outputs/experiments/gfx11-openfold3-model-20261017-v1/GFX11_HANDOFF.before.md`.

## Latest completed follow-up — 2026-10-19

- HIP layout copies passed all three AB/BA/AB pairs and the independent final
  audit. Previous softmax/reuse: **238.181973 s**; added copies: **231.214273 s**;
  **1.030135x incremental** (2.9254% less time). This is not a native-total
  ratio. Forward: 208.234450 -> 201.095808 s.
- All seven accepted predictions match native in all 11 arrays and complete
  CIF bytes. Each candidate: 430 selected triangles, 3,440 HIP copies,
  1,720 native copy fallbacks, unchanged 1,696 HIP softmax submissions.
- Evidence: `provenance/precision_openfold3_copy_gfx1100_20261019.json`.
  Frozen root: `outputs/experiments/gfx11-of3-copy-20261019-v1`; final
  `audit-final.json` and `driver-status.json` passed. Copy tmux jobs completed;
  inspect receipts and current device occupancy before any new GPU job.
- Profiling the preceding composition found pairformer 151.445 s versus diffusion
  rollout 30.265 s (instrumented inclusive intervals). All profiled outputs were
  exact. `provenance/precision_openfold3_profile_gfx1100_20261019.json` records
  raw trace hashes and scope. Main optimization priority remains the trunk.
- Twelve GPU copy cases and six local CPU/meta guard tests passed. Native
  arithmetic and strict FP32/N1000/R3/D200/S1/seed101 are unchanged; original
  maintained workers and earlier immutable experiments were preserved.
- Forward allocator peaks remain 38,636,892,160 allocated / 40,909,144,064 reserved
  bytes. Clocks were unlocked; a sampled SMI record reports THROTTLED.
- Next candidate: reduce attention-score allocation/filling using explicitly
  owned QK scores; the current wave32 ABI rejects aliasing, so a new entry needs
  bitwise operator and full-model gates before timing. C128 normalization is
  another measured cost. Neither follow-up has a new speedup claim.
- General migration remains behind its previous failed AtlasLM gate; no sender
  was resumed or bypassed. Production promotion still needs complete environment
  inventory and publication. N1536 remains separately unqualified on gfx1100.

## Workspace and rules

- Host: `ssh Precision`; workspace: `/media/990Pro/fireprot`.
- GPU: W7900 / Navi 31 / gfx1100. Qualified device: HIP0, PCI `0000:53:00.0`.
- Read `README.md`, `docs/architecture.md`, `.agents/project_context.md`, and
  `docs/precision-setup.md`. Respond to the user in Chinese; write repository
  declarations in English. Continue on `master`.
- Keep checkpoints, outputs and build artifacts outside Git. `Protenix/` is an
  independent upstream checkout. Do not overwrite unrelated user work.

## Verified progress

- Rust CLI/runtime/bridge and HIP identity probe build locally. Pinned Protenix,
  ESM, AtlasFold and OpenFold3 source checkouts are present.
- Python: `/home/husrcf/anaconda3/envs/BIO/bin/python`; Python 3.12.7,
  Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`.
- **ESM3-open native baseline:** four N1000 sequences / B4, native BF16, seed101,
  fresh workers, full hashes/exports, transfers paused during measurement.
  Three E2E times: **12.593587 / 11.883476 / 11.885215 s**;
  median **11.885215 s**. Model-forward median: **2.370584 s**.
- All 12 measured archives match the same-host native reference bitwise.
  Evidence: `provenance/precision_esm3_native_baseline_20261016.json`.
- **ESM3-open has a qualified gfx1100 frameless sequence-only route.** Its original
  three-pair B4/N1000 native-to-candidate E2E comparison is 12.436207 to
  10.953460 seconds median (1.135368x), with exact outputs. Direct GPU loading,
  the snapshot monitor and concurrent source hashing have separate incremental
  evidence; do not multiply ratios or compare timings from different studies.
  HIP Graph, residual fusion and layout follow-ups are not enabled in production.
- **ESM2-3B has a qualified gfx1100 B4/N1000 exact route:** meta/FP16 checkpoint
  loading followed by native FP32 promotion, mmap, exact HIP GEMMs and the
  published K32 bundle. The latest K16-to-K32 comparison is 13.496007 to
  12.528526 seconds median (1.077222x); all outputs match, but one E2E pair
  regresses. This is an incremental result, not a direct-native speedup.
- Maintained pLM requests/profiles and all qualifications are documented in
  `docs/precision-setup.md`. Precision results must remain separate from
  DiamondHill gfx90a results and binaries.

## OpenFold3 / OpenBind-0 full-model qualification

- Source: `runtime/repos/openfold3`, commit
  `7de748b7bc93adb5af0a4032d5e9f208b6e1325f`.
- Python: `runtime/venvs/openfold3-gfx1100/bin/python`, inheriting BIO's Torch.
  The isolated overlay contains the added inference dependencies; BIO was not
  modified. Complete environment-content inventory qualification remains open.
- Checkpoint: `runtime/weights/openbind0/of3-ob-2025-06-30-174k.pt`,
  2,287,872,989 bytes; SHA-256
  `bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4`.
  Source, staging and installed bytes were verified. Receipt:
  `provenance/precision_openfold3_checkpoint_20261017.json`.
- Workload: N1000, strict deterministic FP32 with memory filling, R3/four
  trunk passes, D200/S1, seed101, query-only MSA, no templates, HIP0/PCI53.
- The maintained native entry completed and passed Rust acceptance. Complete
  invocation: **326.240543 s**; model forward: **288.395946 s**. A second native
  invocation using the frozen experimental worker produces all 11 arrays and
  complete CIF bytes identically. The initial invocation pin is
  `c51caca1fa5a5f7845534472925f33e474af108502d4d5d2948072f010f37fc7`.
- Native evidence root:
  `outputs/experiments/gfx11-openfold3-model-20261017-v1`.
- The experimental wave32 softmax library and dispatcher retain native QK/PV
  and source-pinned scoped replacement. Synthetic gates already passed, but
  they are not full-model or E2E performance claims. The maintained gfx90a
  route and compatibility checks are unchanged.
- The first full-model candidate used only the original synthetic shape list.
  Its 8,167 attention calls all fell back to native: zero HIP launches despite
  exact arrays/CIF. The v1 benchmark was deliberately cancelled after the first
  pair; its partial observations are not an optimization result.
- The actual N1000 shapes include outer chunks 256 and 232. The dispatcher now
  admits those chunks for FP32/H4/D32 only. A new 12-case screen covers both
  bias orientations and zero/partial/nonzero masks, with exact complete
  attention and 1.23--1.40x synthetic operator ratios. The 1000-row autotuner
  probes, D16 heads and all other unmatched shapes remain native.
- Current model experiment:
  `outputs/experiments/gfx11-of3-pairs-20261017-v2`, prepared by
  `experiments/hip_softmax/prepare_gfx1100_model.py`; Rust `benchmark-pairs`
  owns execution, exact-array checks and GPU isolation. Independent audit:
  `experiments/hip_softmax/audit_gfx1100_model.py`.
- The frozen experimental worker now rejects successful forwards with zero HIP
  submissions. First pair: 1,696 HIP launches, all 11 arrays and full CIF exact;
  native/candidate E2E 322.522234/287.990005 s, forward 290.049754/254.925590 s.
- **Final three-pair result: passed.** Complete invocation medians are
  **322.388348 s native / 285.978419 s candidate (1.127317x)**. Model-forward
  medians are **290.049754 / 254.290548 s (1.140623x)**. Every pair improves;
  paired ratios are 1.119908, 1.127317 and 1.128889. All six accepted predictions
  match the maintained native reference in all 11 arrays and complete CIF bytes.
  Every candidate submits exactly 1,696 HIP calls. Rust reports
  `faster_median_and_p95`; three-sample p95 is only the observed maximum.
- Evidence: `provenance/precision_openfold3_model_gfx1100_20261017.json`.
  Independent final audit: `outputs/experiments/gfx11-of3-pairs-20261017-v2/audit-final.json`,
  SHA-256 `a709b70bd45ae55b15d5800920f1d366bf7981ba50f6c1f605d27720ad4561c3`.
  Benchmark SHA-256 `7ee2581e0c9e973edb8218a8dcf10dd79760a3f064a846e35da7f38eb8fbd178`.
  The benchmark tmux has exited successfully; do not restart it. The v1 job was
  intentionally cancelled and its records remain separate.
- No N1536 trained-model or production-promotion claim. GPU clocks and caches
  were not fixed; an initial AMD SMI snapshot reported throttling. Do not
  attribute a whole-invocation timing difference solely to the softmax kernel.

## OpenFold3 gfx1100 reuse follow-up (2026-10-18)

- Root: `outputs/experiments/gfx11-of3-reuse-cycles-20261018-v1`. It copies the
  qualified v2 softmax worker; neither that immutable tree nor the maintained
  worker/profile was changed. Strict FP32/N1000/R3/D200/S1/seed101 is unchanged.
- Three cumulative full-model cycles passed all 11 arrays and full CIF bytes
  against the same maintained native reference:

  | Mode | Added treatment | Single E2E | Single forward |
  | --- | --- | ---: | ---: |
  | 1 | Rollout-local pair conditioning | 263.543519 s | 230.005665 s |
  | 2 | Plus static atom representations | 260.604323 s | 228.905371 s |
  | 3 | Plus exact duplicate-template slots | 241.414822 s | 208.841154 s |

- These individual timings are diagnostics, not repeated ablation speedups.
  Each enabled pair/atom cache records 200 calls, one miss, 199 hits and zero
  fallbacks. Mode 3 records nine eligible template calls and 27 reused slots.
  The no-template input still has internal padded slots; distinct slot/mask
  bits, including signed-zero differences, retain native execution.
- Pair/atom reuse is scoped to a single rollout and requires the pinned
  immutable-input source contract. Identity keys do not detect arbitrary
  in-place changes. Time-dependent single conditioning and noisy-coordinate
  projection are recomputed at every step; the native template reduction stays.
- All three cycle allocator peaks are 38,636,892,160 allocated and
  40,909,144,064 reserved bytes, reset just before forward. Pair/atom tensors
  retained by caches are 512,000,000 / 69,620,736 bytes; these are not additive
  whole-model peak costs. No hardware-memory or whole-invocation peak claim.
- The copied N1000 template guard passes all six boundary/fallback checks.
  Composition tests cover all four modes, unsupported devices, wrong hit counts,
  zero template coverage and exception restoration.
- **Final three-pair AB/BA/AB comparison: passed.** Previous softmax-only versus
  combined reuse invocation medians are **283.614466 / 239.334872 s (1.185011x,
  15.6126% less time)**. Forward medians are **252.733066 / 208.135715 s
  (1.214271x)**. All three pairs improve: 1.190806, 1.183716 and 1.193164x.
  This is an incremental result, not a direct native speedup; never multiply it
  by the preceding softmax study's ratio.
- All nine accepted predictions (three cycles plus six paired runs) match all
  11 arrays and complete CIF bytes against the original maintained native run.
  Every run submits 1,696 HIP softmax calls. All six paired allocator peaks equal
  the cycle peaks above; no peak-memory reduction is claimed. Fifty CPU tests
  pass: 21 conditioning, 12 composition and 17 maintained worker/scope tests.
- Evidence: `provenance/precision_openfold3_reuse_gfx1100_20261018.json`.
  Final audit SHA-256:
  `ef510e390565fc63dc829e727b55435f88994091ecef66c2d74e027a4cff9039`.
  Benchmark SHA-256:
  `b8681756cb67a4bbf16b52eb5099b93acc7b9a2efb4e9d1ea83b99011d4d6919`.
  All three cycle tmux sessions and `fireprot-of3-reuse-pairs` have completed;
  inspect durable receipts instead of restarting them.
- Caches were unflushed, clocks unlocked, and the retained cycle-3 SMI sample
  reports `THROTTLED`. Three-sample p95 is the observed maximum, not a confidence
  bound. Complete environment inventory and raw-kernel publication remain open;
  neither production promotion nor N1536 qualification is claimed.
- Reproduction and exact treatment declarations:
  `experiments/hip_softmax/README.md`, `prepare_gfx1100_reuse.py`,
  `prepare_gfx1100_reuse_pairs.py`, `audit_gfx1100_reuse.py` and
  `collect_gfx1100_reuse.py` in that directory. Never reuse an output directory.

## Native entries

```bash
cd /media/990Pro/fireprot
cp configs/requests/precision-esm3-open-b4-n1000-native.json /tmp/my-esm3.json
# Set a fresh run_id and unused output_dir in /tmp/my-esm3.json.
target/release/fireprot esm /tmp/my-esm3.json configs/precision/plm-esm3-open-native.json
```

OpenFold3 uses the separately prepared venv/profile:

```bash
cd /media/990Pro/fireprot
cp configs/requests/precision-openfold3-n1000-native.json /tmp/my-openfold3.json
# Set a fresh run_id and unused output_dir in /tmp/my-openfold3.json.
target/release/fireprot infer /tmp/my-openfold3.json configs/precision/openfold3-native.json
```

For the experimental optimized worker, use the frozen preparation procedure in
`experiments/hip_softmax/README.md`; do not simply reuse the gfx90a HIP profile.

## Background migration: do not duplicate jobs

The user selected Wi-Fi and detached tmux operation. Manifest: **49 files,
52,778,669,198 bytes**. Existing ESM2/ESM3/ESMFold-v1 weights were reused.
At the checkpoint-transfer snapshot, **22 files remain pending by existence/size checks**;
final all-file SHA-256 verification is not complete.

| Host | tmux session | Role |
| --- | --- | --- |
| DiamondHill (`ssh DiamondHill`) | `fireprot-refresh-finalize` | Exited with `AssertionError('atlaslm-b4-refresh')`; general release was not granted |
| DiamondHill | `fireprot-precision-transfer` | Wait for release, then transfer missing files with partial-file retries |
| Precision | `fireprot-assets-verify` | Wait for all files, then verify complete SHA-256 hashes |

DiamondHill logs/status: `/media/IntelSSD/fireprot/outputs/experiments/e2e-refresh-20261016/`:
`finish-and-release.log`, `finish-and-release-status.json`,
`asset-transfer-tmux.log`, `asset-transfer-v2-status.json`.
Precision logs/status: `runtime/migration-20261016/`:
`verification-progress.json`, `verification-tmux.log`, `verified-assets.json`.
Only the last file with `status=all_files_sha256_verified` proves completion.
The user separately authorized the single OpenFold3 checkpoint transfer. That
transfer completed; it did not bypass or modify the general migration gate.
The old rsync had verified other assets, but had never copied this checkpoint;
there was no evidence of a previously completed checkpoint becoming corrupt.
Completed tmux sessions exit; inspect durable logs. Never bypass a failed gate
or restart the exclusive Base collector blindly. DiamondHill tests use HIP6
serially; do not add overlapping jobs there.

## Next work

1. Inspect durable OpenFold3 receipts and any existing tmux job before launching
   another GPU workload. Never overlap runs on HIP0.
2. N1000 full-model and three-pair gates now pass for both frozen softmax and
   the subsequent pair/atom/template reuse composition.
   Before enabling a maintained gfx1100 OpenFold3 profile, publish the kernel
   through Rust, complete the environment inventory, and requalify the packaged
   route with actual HIP coverage and all-array/CIF checks.
3. Keep exact precision, workload, source/checkpoint hashes and native fallbacks.
   Do not port gfx90a binaries, turn on Triton, or relax deterministic gates.
4. Qualify a complete OpenFold3 environment inventory and separately test N1536
   before broadening support. Preserve the independent migration gate and jobs.
5. After DiamondHill's final collector succeeds, update README and `.summary.md`
   with all accepted fresh model ratios. Keep Precision results in a separate table.
