# GitNexus Engineering Plan

> Task: OpenFold3 workspace integration, revised after exact model gates exposed low coverage.
> Evidence verified at `658c4b0ad4d509ee3a6e1fd3f3d708a3fecde7a9`; GitNexus1.6.12 typed index/runner current, no refresh needed. MCP impact's behind59 label is inconsistent with typed receipts.
> Evidence provenance schema2; global dirty digest `2c6402b0053bf1b1d85ccb785f269c24e0d3531ff1b0a288fad3cf5b9ec36643`; 19 sorted cited paths. Existing v1 implementation is complete through correctness gates; capacity/coverage and E2E remain open.

## 1. Objective

Recover useful full-model attention coverage with a smaller, explicitly qualified single workspace, preserving exact scientific outputs and measuring E2E only after the coverage issue is resolved.

## 2. Current Behaviour

[verified] Steps1-5 of the original implementation are present: Rust lease and bridge, one B256 tensor owner, opt-in dispatcher/worker patch, mixed-batch GPU qualification, and N1000/N1536 exact model gates. Evidence: `provenance/openfold3_model_workspace_model_gates.json`.

[verified] N1536 matches all11 arrays and complete CIF bytes against both original native and accepted D16 references. However, MSA/Pairformer chunks changed1024 to512, the fallback now records5,040 B128 calls, and the9GiB owner serves only28 full calls. N1000 matches native and allocates no pool. These overlapping correctness runs are not latency evidence. No v1 paired timing was started.

[verified] `ModelWorkspaceFusion` fixes shape/budget at lines15-25 and loop/direct-output choices at95-105. `WorkspaceFusion.__init__` accepts32/256; the loader requires21 exact numerical-screen records for its precise batch. `load_workspace` verifies the control bytes before creating an owner.

## 3. Relevant Architecture

[verified] `crates/fireprot-core/src/registry.rs:13` defines metadata signatures; selection is cached by signature. `python/fireprot/ops/registry.py:1` retains tensor callbacks around that Rust policy. `crates/fireprot-bridge/src/lib.rs:148` registers the bridge module. Core must remain independent of Torch/GPU calls.

[verified] `python/fireprot/adapters/openfold3/attention.py:30` supplies a scoped hook; `python/fireprot_workers/openfold3.py:116` scopes backend dispatch and writes its runtime manifest in a finally path. The model trial must overlay the accepted experimental D16 deployment, including its conditioning caches, owned-score softmax and confidence cap; the maintained base worker alone is not that baseline.

## 4. GitNexus Findings

[graph] At658c4b0, `impact(ModelWorkspaceFusion, upstream, depth3, includeTests)` reports LOW, four direct entries: qualifier file/main, dispatcher and model-owner tests; dispatcher tests are depth2. Preserve all these consumers when adding a capacity parameter. Clusters/processes still separate experiment, operator and worker concerns.

[verified] Typed CLI and context agree on current HEAD, current schema4 runner and empty incomplete reasons. The impact response's behind59 label disagrees; source and typed receipts take precedence. Dynamic upstream hooks are outside the indexed caller proof.

## 5. Statement-Level PDG Findings

[graph] Bounded `pdg_query(controls, workspace_attention.py, limit=200)` returns114 edges. The relevant slice is `__call__`63–104 and `close`106–115. `flows(variable=self._scores)` returns zero edges; attribute data flow is not established by that result.

[verified] Source ordering is: lock64; guards68–79; first allocation80–85; mark submission86; HIP87–90; status guard91; PV93; private-storage guard94; successful-call counter97; poison on submission exception100–101; unlock104. Cleanup guards107, synchronizes111, then drops112. The plan preserves this ordering, including native PV before any same-stream overwrite. Extend model-boundary cleanup to interruptions as well as ordinary exceptions. Do not infer Python exception semantics solely from graph branch labels.

[graph] A current bounded PDG controls query for `model_workspace.py` returns125 edges. [verified] Source confirms first allocation91-92, begin_use93, loop95, native pool call99, finish_use106 and failure poisoning110-112. Capacity changes must update constructor shape, allocation, loop stride, last slice and direct-result handling coherently.

[verified] External pinned `ChunkSizeTuner.tune_chunk_size` caches shape/itemsize argument metadata; `_determine_favorable_chunk_size` runs its representative function under no_grad and catches every RuntimeError. [inferred] Memory pressure is plausible but not diagnosed until failed-call exceptions are recorded.

## 6. Proposed Changes

1. [inferred] Add NEW `tuner_observer` in `experiments/mfma_qk/tuner_observer.py`. Verify the pinned upstream source, wrap the representative function passed to the unchanged tuner, record candidate size, outcome and exception type/message, then rethrow the identical exception. Preserve args/cache behavior, return values and thread/context scope. Restore on normal/exceptional exit; reject conflicting scopes or foreign hook mutation.
2. [inferred] Extend `model-workspace.patch` with an explicit diagnostic flag and report. First run the unchanged9GiB configuration to identify the actual rejection. Diagnostics are disabled in final latency measurements.
3. [inferred] Parameterize `ModelWorkspaceFusion` with fixed per-forward capacity32,128 or256; keep256 as the v1 reproduction default. Derive score shape and exact byte budget (1.125,4.5 or9GiB) from capacity. Keep one pool/lease. Permit only qualified, divisible outer batches from128/256/512/1024; unmatched shapes retain accepted fallback. No padding or unverified tail handling.
4. [inferred] Qualify combined-v12 at128 with the existing21-case raw/softmax screen before enabling that pool. Update `WorkspaceFusion.__init__` accordingly. Capacity32 already has a kernel screen but still needs full-reference microbatch/model qualification.
5. [inferred] Make `load_workspace` read capacity from hashed controls; make the mixed-batch qualifier derive expected allocations, bytes and launch counts. Compare complete operator cost for smaller candidates, then repeat model checks and record actual coverage/tuners.
6. [inferred] Select an evidence-supported capacity for isolated paired E2E. Preserve strict filling, FP32, R3/D200/S1, seed101, D16 behavior, caches and confidence policy. No HIP arithmetic change is planned.

## 7. Implementation Sequence

Completed at658c4b0: original steps1-5 (core, bridge, owner, mixed GPU and exact model gates). Skip their implementation; retain them as regressions. Original paired timing was deliberately not started because coverage collapsed.

1. Add observer, exception/cache/restoration tests and opt-in worker reporting. Reproduce the unchanged9GiB run with diagnostics; verify all11 arrays/CIF again.
2. Verify the B12821-case kernel screen; add explicit capacity handling and CPU transition/dispatch tests. Independent kernel qualification may run before the diagnostic model, with no competing timed workload.
3. Qualify full probability/PV bits, old outputs, masks/layouts and one allocation across measured outer batches for each candidate. Screen complete attention cost, including packing, bias normalization and assembly.
4. Run N1536 candidates with diagnostic rejection/coverage records; resolve any non-memory errors before selecting capacity. Preserve all11-array/CIF gates. Run N1000 fallback regression for the selected implementation.
5. Only after useful coverage and model gates, run isolated three-pair AB/BA/AB against the accepted D16 baseline, same rebuilt framework/worker and diagnostics disabled. Record cache state, actual chunks and peaks. Refresh a direct original-Torch comparison only if promotion is justified; never multiply historical speedup ratios.
6. Archive results and decide promotion or experimental rejection. No plateau is established yet.

## 8. Test Strategy

The context pack below specifies new state and tensor-boundary scenarios. Existing `tests/operators/test_openfold3_workspace.py`, `test_openfold3_microbatch.py`, `test_openfold3_bias_layouts.py` and `test_rust_registry.py` are regression gates. Build the bridge before importing the shared operator API. The verified Cargo workspace supports:

```bash
cargo test --workspace --locked
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
python -m pytest -q tests/operators/test_rust_registry.py tests/operators/test_openfold3_workspace.py tests/operators/test_openfold3_microbatch.py tests/operators/test_openfold3_bias_layouts.py
```

Run new test files when added. GPU commands must be generated from pinned deployment manifests; no invented runnable benchmark command is provided here. Gate model output before performance trials, and repeat correctness for every timed model run.

New observer tests must demonstrate unchanged selection and the same caught exception object; capacity tests must use the real Rust lease. Add the new test files to the step-relevant pytest command. Existing180 Rust tests and51 Python tests passed before this revision; their results do not validate future changes.

## 9. Risk and Impact Analysis

[verified] Old shape frequencies are not a reliable future coverage guarantee: the retained pool coincided with a tuner shift. [inferred] Capturing rejection types is required to distinguish memory pressure from a guard or unrelated RuntimeError. Do not silently force a chunk or remove a guard.

[graph] All direct ModelWorkspaceFusion consumers are the qualifier (file/main), dispatcher and model-owner tests; update them together. Existing fixed-wrapper benchmarks retain their256 behavior. [inferred] Smaller capacities trade lower residency against additional launches and concatenation; operator throughput alone cannot establish E2E benefit. The observer must restore the upstream method and preserve cached-call behavior.

## 10. Files Expected to Change

| File | Responsibility |
| --- | --- |
| `experiments/mfma_qk/tuner_observer.py` | Source-pinned scoped observer that preserves upstream exception and cache behavior |
| `experiments/mfma_qk/model-workspace.patch` | Add diagnostic scope/reporting to the frozen-worker overlay; keep original base hash |
| `experiments/mfma_qk/model_workspace.py` | Explicit fixed microbatch parameter with derived shape/bytes and divisible measured batch eligibility |
| `experiments/mfma_qk/workspace_attention.py` | Allow128 only with the existing full pinned numerical screen gate |
| `experiments/mfma_qk/workspace_dispatch.py` | Load microbatch from hashed controls, preserving default256 reproduction |
| `experiments/mfma_qk/qualify_model_workspace.py` | Capacity-aware full-reference gates, exact allocation and call accounting |
| `experiments/mfma_qk/WORKSPACE.md` | Record diagnostics, qualification, coverage, memory and correctly scoped timing |

Tests and evidence artifacts are listed in §11. Rust core/bridge policy is already implemented and requires no capacity-specific rewrite.

## 11. Reusable Implementation Context

```json
{
  "implementation_context": {
    "task_summary": "Diagnose tuner rejection causes, qualify smaller single-pool capacities and B128 coverage, then establish isolated model E2E benefit.",
    "acceptance_criteria": [
      "One Rust-governed pool per forward; no independent pools per outer batch",
      "Capture tuner RuntimeError causes without changing upstream selection or exception identity",
      "Qualify only explicit capacities32/128/256 and divisible measured outer batches128/256/512/1024; retain256 for v1 reproduction",
      "Strict FP32 R3/D200/S1 seed101 and ordinary memory filling remain enabled",
      "All11 arrays bitwise and CIF identity for every model trial",
      "Demonstrated model coverage plus isolated AB/BA/AB E2E before promotion"
    ],
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "658c4b0ad4d509ee3a6e1fd3f3d708a3fecde7a9",
      "generated_plan_path": "docs/plans/2026-10-06-gitnexus-plan-openfold3-workspace-model-integration.md",
      "global_dirty_digest": {
        "algorithm": "sha256",
        "canonicalization": "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records",
        "value": "2c6402b0053bf1b1d85ccb785f269c24e0d3531ff1b0a288fad3cf5b9ec36643"
      },
      "cited_path_manifest": [
        {
          "path": "Cargo.toml",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:cd43e29c029b7a807e6fa97a0ed5dc2c43127ca4e672b88a3a97afc7a814b806",
          "index_digest": "sha256:cd43e29c029b7a807e6fa97a0ed5dc2c43127ca4e672b88a3a97afc7a814b806",
          "worktree_digest": "sha256:cd43e29c029b7a807e6fa97a0ed5dc2c43127ca4e672b88a3a97afc7a814b806",
          "untracked_digest": "absent"
        },
        {
          "path": "crates/fireprot-bridge/src/lib.rs",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:962e48eec1a7a477a552da737f536e4446619ffbb1c6bb5e6adb8895d31bb644",
          "index_digest": "sha256:962e48eec1a7a477a552da737f536e4446619ffbb1c6bb5e6adb8895d31bb644",
          "worktree_digest": "sha256:962e48eec1a7a477a552da737f536e4446619ffbb1c6bb5e6adb8895d31bb644",
          "untracked_digest": "absent"
        },
        {
          "path": "crates/fireprot-core/src/lib.rs",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:af245f606d1c95c8de7d0cd5302db9fd9c48279be2b653c6ca8efb94f5c9e893",
          "index_digest": "sha256:af245f606d1c95c8de7d0cd5302db9fd9c48279be2b653c6ca8efb94f5c9e893",
          "worktree_digest": "sha256:af245f606d1c95c8de7d0cd5302db9fd9c48279be2b653c6ca8efb94f5c9e893",
          "untracked_digest": "absent"
        },
        {
          "path": "crates/fireprot-core/src/registry.rs",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:e8126028302455022bc4640445b3773cc5f118de4da78f163b6c9c67d0b71a56",
          "index_digest": "sha256:e8126028302455022bc4640445b3773cc5f118de4da78f163b6c9c67d0b71a56",
          "worktree_digest": "sha256:e8126028302455022bc4640445b3773cc5f118de4da78f163b6c9c67d0b71a56",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/microbatch_attention.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:ededef982e850fd3aad17bb6861ac4cbec5ea58fef99105fd94f6372466f1b01",
          "index_digest": "sha256:ededef982e850fd3aad17bb6861ac4cbec5ea58fef99105fd94f6372466f1b01",
          "worktree_digest": "sha256:ededef982e850fd3aad17bb6861ac4cbec5ea58fef99105fd94f6372466f1b01",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/model-workspace.patch",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:537038b2aaa1c13cfccd163e5b36cbfbc68d0ed64369eaa382ed2e4350656e59",
          "index_digest": "sha256:537038b2aaa1c13cfccd163e5b36cbfbc68d0ed64369eaa382ed2e4350656e59",
          "worktree_digest": "sha256:537038b2aaa1c13cfccd163e5b36cbfbc68d0ed64369eaa382ed2e4350656e59",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/model_workspace.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:c604369d2106f70625bbdd81bc8aea3df5b714e2d175ad2219c81490cd200ae8",
          "index_digest": "sha256:c604369d2106f70625bbdd81bc8aea3df5b714e2d175ad2219c81490cd200ae8",
          "worktree_digest": "sha256:c604369d2106f70625bbdd81bc8aea3df5b714e2d175ad2219c81490cd200ae8",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/qualify_model_workspace.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:be9d1951d968c065f31dc910e3619c6d246fcb26bc9d28077adf90e74e6fd807",
          "index_digest": "sha256:be9d1951d968c065f31dc910e3619c6d246fcb26bc9d28077adf90e74e6fd807",
          "worktree_digest": "sha256:be9d1951d968c065f31dc910e3619c6d246fcb26bc9d28077adf90e74e6fd807",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/workspace_attention.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:eb3883cf778f2ea85b9e150b11f556cc65e97ec0d3210a67c5925efeeec57a24",
          "index_digest": "sha256:eb3883cf778f2ea85b9e150b11f556cc65e97ec0d3210a67c5925efeeec57a24",
          "worktree_digest": "sha256:eb3883cf778f2ea85b9e150b11f556cc65e97ec0d3210a67c5925efeeec57a24",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/workspace_dispatch.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:81f46174e6b612815172f7273b48fa719fe65d36c977fa4855918ceb3c2b6406",
          "index_digest": "sha256:81f46174e6b612815172f7273b48fa719fe65d36c977fa4855918ceb3c2b6406",
          "worktree_digest": "sha256:81f46174e6b612815172f7273b48fa719fe65d36c977fa4855918ceb3c2b6406",
          "untracked_digest": "absent"
        },
        {
          "path": "provenance/openfold3_model_workspace_model_gates.json",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:30d5e52e34d74d1064a8868bb21ebd27fea475ea80d42c15f1d6cb6755ab10fa",
          "index_digest": "sha256:30d5e52e34d74d1064a8868bb21ebd27fea475ea80d42c15f1d6cb6755ab10fa",
          "worktree_digest": "sha256:30d5e52e34d74d1064a8868bb21ebd27fea475ea80d42c15f1d6cb6755ab10fa",
          "untracked_digest": "absent"
        },
        {
          "path": "python/fireprot/adapters/openfold3/attention.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:d8a884b9c018c436904a1e6774139e75915fae8c11189190425865a5780b9bd3",
          "index_digest": "sha256:d8a884b9c018c436904a1e6774139e75915fae8c11189190425865a5780b9bd3",
          "worktree_digest": "sha256:d8a884b9c018c436904a1e6774139e75915fae8c11189190425865a5780b9bd3",
          "untracked_digest": "absent"
        },
        {
          "path": "python/fireprot/ops/registry.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:82cdb1cb2892d7571b765bde564651dcb92650c8350675e757e8359927419c8f",
          "index_digest": "sha256:82cdb1cb2892d7571b765bde564651dcb92650c8350675e757e8359927419c8f",
          "worktree_digest": "sha256:82cdb1cb2892d7571b765bde564651dcb92650c8350675e757e8359927419c8f",
          "untracked_digest": "absent"
        },
        {
          "path": "python/fireprot_workers/openfold3.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e",
          "index_digest": "sha256:c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e",
          "worktree_digest": "sha256:c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_openfold3_microbatch.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:ecab5da22d574e15703549e58147e006b8736754d622bcf832cfae40516b0751",
          "index_digest": "sha256:ecab5da22d574e15703549e58147e006b8736754d622bcf832cfae40516b0751",
          "worktree_digest": "sha256:ecab5da22d574e15703549e58147e006b8736754d622bcf832cfae40516b0751",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_openfold3_model_workspace.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:6e01381133dfd346aaf2f0c6a71ec3701c300ac8d180829e44c662125924b4ce",
          "index_digest": "sha256:6e01381133dfd346aaf2f0c6a71ec3701c300ac8d180829e44c662125924b4ce",
          "worktree_digest": "sha256:6e01381133dfd346aaf2f0c6a71ec3701c300ac8d180829e44c662125924b4ce",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_openfold3_workspace.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:be53f8dee130afbb59ea132ceb5f8fd186d3039a56ece9036b83b84b3153ca53",
          "index_digest": "sha256:be53f8dee130afbb59ea132ceb5f8fd186d3039a56ece9036b83b84b3153ca53",
          "worktree_digest": "sha256:be53f8dee130afbb59ea132ceb5f8fd186d3039a56ece9036b83b84b3153ca53",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_openfold3_workspace_dispatch.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:20d99904a6dcc7ae790fa83eaf745d8419458414ae3656c5dacd1913d8bf49e8",
          "index_digest": "sha256:20d99904a6dcc7ae790fa83eaf745d8419458414ae3656c5dacd1913d8bf49e8",
          "worktree_digest": "sha256:20d99904a6dcc7ae790fa83eaf745d8419458414ae3656c5dacd1913d8bf49e8",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_rust_registry.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:56b81c26042ad8defd769925a5d5f6bb2ea6d33bb3cedb997743680444ccc7ac",
          "index_digest": "sha256:56b81c26042ad8defd769925a5d5f6bb2ea6d33bb3cedb997743680444ccc7ac",
          "worktree_digest": "sha256:56b81c26042ad8defd769925a5d5f6bb2ea6d33bb3cedb997743680444ccc7ac",
          "untracked_digest": "absent"
        }
      ]
    },
    "primary_symbols": [
      {
        "symbol": "ModelWorkspaceFusion",
        "file": "experiments/mfma_qk/model_workspace.py",
        "lines": "19-158",
        "role": "Parameterize one fixed-capacity owner without changing lifecycle ordering"
      },
      {
        "symbol": "WorkspaceFusion.__init__",
        "file": "experiments/mfma_qk/workspace_attention.py",
        "lines": "25-37",
        "role": "Gate newly qualified128 capacity"
      },
      {
        "symbol": "load_workspace",
        "file": "experiments/mfma_qk/workspace_dispatch.py",
        "lines": "12-16",
        "role": "Read capacity from hashed experimental controls"
      },
      {
        "symbol": "main",
        "file": "experiments/mfma_qk/qualify_model_workspace.py",
        "lines": "19-142",
        "role": "Verify mixed capacities and full native references"
      }
    ],
    "related_symbols": [
      {
        "symbol": "ChunkSizeTuner.tune_chunk_size",
        "relationship": "external pinned upstream seam",
        "relevance": "Pass a recording representative_fn wrapper to unchanged original method; preserve args, cache and exception behavior"
      },
      {
        "symbol": "WorkspaceDispatcher",
        "relationship": "imports ModelWorkspaceFusion",
        "relevance": "Metadata eligibility must reflect the chosen fixed capacity; keep accepted fallback intact"
      }
    ],
    "execution_path": [
      "Verify pinned upstream tuner source and install a scoped diagnostic wrapper",
      "Execute original tuning method with unchanged args and cache; record candidate failures then rethrow identical exceptions",
      "First diagnose the unchanged9GiB configuration",
      "Qualify smaller fixed capacities independently; allocate exactly one ordinary strictly initialized score pool",
      "Process measured outer batches in capacity-sized slices; preserve same-stream native PV ordering",
      "Drain/drop/acknowledge release; compare all model outputs before isolated timing"
    ],
    "pdg_constraints": [
      {
        "description": "First score allocation and launch have distinct failure boundaries",
        "affected_statements": [
          "experiments/mfma_qk/workspace_attention.py:80",
          "experiments/mfma_qk/workspace_attention.py:86",
          "experiments/mfma_qk/workspace_attention.py:100"
        ],
        "implementation_consequence": "Reservation rollback is only valid before allocation commitment; postsubmission failure poisons the owner."
      },
      {
        "description": "Close must drain before releasing score storage",
        "affected_statements": [
          "experiments/mfma_qk/workspace_attention.py:107",
          "experiments/mfma_qk/workspace_attention.py:111",
          "experiments/mfma_qk/workspace_attention.py:112"
        ],
        "implementation_consequence": "A failed synchronization must not produce released state or zero retained bytes."
      },
      {
        "description": "Current model owner allocates once at91-92, begins Rust use93, then loops95 before finish106",
        "affected_statements": [
          "experiments/mfma_qk/model_workspace.py:91",
          "experiments/mfma_qk/model_workspace.py:95",
          "experiments/mfma_qk/model_workspace.py:106"
        ],
        "implementation_consequence": "Change all capacity-dependent shape/range/direct-result decisions together; never allocate a pool per outer shape."
      }
    ],
    "architectural_patterns": [
      {
        "pattern": "Rust owns metadata policy; Python owns tensor calls",
        "example_location": "crates/fireprot-bridge/src/lib.rs",
        "usage_guidance": "Add a lease class; do not claim Rust verifies GPU completion independently."
      },
      {
        "pattern": "Scoped upstream attention hook",
        "example_location": "python/fireprot/adapters/openfold3/attention.py:30",
        "usage_guidance": "Keep hook ownership/restoration and preserve the Apache-2.0 attribution on derived material."
      }
    ],
    "files_to_modify": [
      {
        "file": "experiments/mfma_qk/tuner_observer.py",
        "symbols": [
          "NEW tuner_observer"
        ],
        "intended_change": "Source-pinned scoped observer that preserves upstream exception and cache behavior"
      },
      {
        "file": "experiments/mfma_qk/model-workspace.patch",
        "symbols": [],
        "intended_change": "Add diagnostic scope/reporting to the frozen-worker overlay; keep original base hash"
      },
      {
        "file": "experiments/mfma_qk/model_workspace.py",
        "symbols": [
          "ModelWorkspaceFusion"
        ],
        "intended_change": "Explicit fixed microbatch parameter with derived shape/bytes and divisible measured batch eligibility"
      },
      {
        "file": "experiments/mfma_qk/workspace_attention.py",
        "symbols": [
          "WorkspaceFusion.__init__"
        ],
        "intended_change": "Allow128 only with the existing full pinned numerical screen gate"
      },
      {
        "file": "experiments/mfma_qk/workspace_dispatch.py",
        "symbols": [
          "load_workspace"
        ],
        "intended_change": "Load microbatch from hashed controls, preserving default256 reproduction"
      },
      {
        "file": "experiments/mfma_qk/qualify_model_workspace.py",
        "symbols": [
          "main"
        ],
        "intended_change": "Capacity-aware full-reference gates, exact allocation and call accounting"
      },
      {
        "file": "experiments/mfma_qk/WORKSPACE.md",
        "symbols": [],
        "intended_change": "Record diagnostics, qualification, coverage, memory and correctly scoped timing"
      }
    ],
    "tests": [
      {
        "file": "tests/operators/test_openfold3_tuner_observer.py (new)",
        "scenarios": [
          "Representative function raises RuntimeError -> log exact type/message and rethrow the same object; upstream chooses the same chunk",
          "Cached tuner result -> no extra representative function execution",
          "Normal exit, model exception and interruption -> restore original hook",
          "Source-hash mismatch, concurrent/nested scope, foreign hook mutation -> reject visibly",
          "Original args and metadata remain unchanged"
        ]
      },
      {
        "file": "tests/operators/test_openfold3_model_workspace.py",
        "scenarios": [
          "Capacities32/128/256 -> exact derived shape/budget, one allocation, correct slice counts",
          "Nondivisible or unqualified outer shape -> reject before allocation",
          "All existing submission, cleanup, interruption and owner guards remain enforced"
        ]
      },
      {
        "file": "tests/operators/test_openfold3_workspace_dispatch.py",
        "scenarios": [
          "Hashed control chooses capacity; missing capacity preserves256",
          "Cached registry selection still executes dynamic lease guards",
          "Unmatched signatures retain accepted D16/native fallback"
        ]
      },
      {
        "file": "tests/operators/test_openfold3_workspace.py",
        "scenarios": [
          "Budget rejects insufficient bytes and unsupported capacity before loader; new128 still requires pinned numerical screen"
        ]
      }
    ],
    "verification_commands": [
      "cargo test --workspace --locked",
      "cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked",
      "python -m pytest -q tests/operators/test_rust_registry.py tests/operators/test_openfold3_workspace.py tests/operators/test_openfold3_microbatch.py tests/operators/test_openfold3_bias_layouts.py"
    ],
    "risks": [
      "Upstream catches all RuntimeError, so a smaller cached chunk is not proof of OOM",
      "Extra retained memory can shift actual batch shapes and invalidate old coverage assumptions",
      "Smaller microchunks add packing/concatenation/launch costs; time complete operators and models",
      "Observer must not alter exception identity, tuner selection or cached argument data"
    ],
    "assumptions": [
      "Unchanged9GiB diagnostic run reproduces the tuner shift; verify actual rejection records before assigning a cause.",
      "Combined-v12 bundle remains unchanged; verify manifest1f04b110c79520749dc2a9a483b77f35330b559f5c043dfcdd13c72947ec5122.",
      "Pinned upstream tuner source remains SHA4c9ab484854868e3c3d9a28a0d2e8fe0f9b1c75a35dca4984e0631591489fac5."
    ],
    "open_questions": [
      "Which RuntimeError caused the tuner to reject larger chunks?",
      "Which qualified capacity best preserves model coverage and reduces E2E?",
      "Does128 or32 capacity restore larger chunks, and at what complete-operator cost?"
    ],
    "avoid": [
      "Do not repeat full repository discovery",
      "Do not create multiple pools per model forward or silently broaden unqualified shapes",
      "Do not disable strict deterministic execution or uninitialized-memory filling",
      "Do not change HIP arithmetic, reduction order, precision, recycle count or diffusion steps",
      "Do not fall back after partial GPU submission",
      "Do not multiply incremental speedup ratios into a direct original-Torch claim",
      "Do not commit docs/isa/CDNA2.pdf",
      "Do not launch v1 paired timing for the28-call coverage result",
      "Do not force the old chunk size or swallow additional exceptions",
      "Do not interpret concurrent correctness runs as isolated timing"
    ]
  }
}
```

## 12. Assumptions and Open Questions

[assumed] An isolated9GiB diagnostic run reproduces the tuner shift. Inspect actual rejected-call exceptions; do not assume OOM from cached chunks alone. [assumed] The bundle and external tuner source retain the hashes in §11; verify before staging.

Open: actual rejection cause, capacity/coverage/latency tradeoff, and whether the smaller owner restores larger chunks. B64-specific optimization, other sequence lengths, arithmetic changes and generalized cross-model resource services remain deferred. New diagnostics are a measurement aid, not a replacement tuner.

## 13. Definition of Done

- Exact exception diagnostics establish why larger chunks were rejected, with unchanged tuner selection semantics.
- One explicitly qualified pool capacity serves the observed target shapes and has truthful allocation/release accounting.
- All model trials preserve all11 bitwise arrays and CIF identity under the fixed scientific contract; unmatched cases retain validated fallback.
- Isolated paired E2E, coverage, tuning and memory evidence supports promotion, or records a measured rejection. Correctness-only elapsed times remain excluded.
- Artifacts/source/binaries/inputs/weights/cache state are pinned and archived. No unmeasured gain or overall speedup inferred by multiplication.
