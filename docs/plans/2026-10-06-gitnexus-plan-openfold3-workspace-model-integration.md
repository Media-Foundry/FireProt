# GitNexus Engineering Plan

> Task: Integrate the measured OpenFold3 attention workspace into an opt-in model run.
> Evidence verified at commit `34e674ab4b82da83e35e08388ea9782b1c892973`; index refreshed this session with `--index-only --pdg`, GitNexus 1.6.12, current runner identity and source content.
> Evidence provenance schema 2; global dirty digest `2c6402b0053bf1b1d85ccb785f269c24e0d3531ff1b0a288fad3cf5b9ec36643`; 12 sorted cited paths; only this generated plan path excluded.
> MCP impact reports “52 commits behind” despite matching resource/CLI index and HEAD. Source and typed CLI status take precedence; graph call coverage is not proof of dynamic Python coverage.

## 1. Objective

Integrate the already measured HIP QK/softmax implementation through one bounded per-forward workspace, then establish correctness and actual E2E benefit. This plan introduces no new arithmetic. No workspace model result has yet passed.

## 2. Current Behaviour

- [verified] `WorkspaceFusion.__call__`, `experiments/mfma_qk/workspace_attention.py:63`, locks, checks strict settings and ownership, packs Q/K, optionally normalizes bias, lazily allocates scores, launches HIP and runs native PV. Only the PV result escapes.
- [verified] `WorkspaceFusion.close`, line106, synchronizes the bound device before dropping the scores. The existing class is experimental and has no Rust resource lease.
- [verified] `MicrobatchFusion`, `experiments/mfma_qk/microbatch_attention.py:1`, owns its own fixed B256 pool for a fixed outer B512 or B1024. Creating both wrappers plus a direct pool would retain 27 GiB.
- [verified] Existing CPU tests cover budget, thread/lock, strict-setting, lazy-view and closed-instance rejection; they do not establish model equivalence.
- [inferred] Reusing one initialized score tensor can remove repeated strict-fill cost while keeping strict settings enabled. Operator gains are not E2E evidence.

## 3. Relevant Architecture

[verified] `crates/fireprot-core/src/registry.rs:13` defines metadata signatures; selection is cached by signature. `python/fireprot/ops/registry.py:1` retains tensor callbacks around that Rust policy. `crates/fireprot-bridge/src/lib.rs:148` registers the bridge module. Core must remain independent of Torch/GPU calls.

[verified] `python/fireprot/adapters/openfold3/attention.py:30` supplies a scoped hook; `python/fireprot_workers/openfold3.py:116` scopes backend dispatch and writes its runtime manifest in a finally path. The model trial must overlay the accepted experimental D16 deployment, including its conditioning caches, owned-score softmax and confidence cap; the maintained base worker alone is not that baseline.

## 4. GitNexus Findings

- [graph] Context/clusters/processes resources identify Mfma_qk, Operators, Fireprot_workers and Adapters as separate areas. Those resources guide boundaries, not runtime ownership claims.
- [graph] `impact(target=WorkspaceFusion, direction=upstream, maxDepth=3)` reports **MEDIUM**, seven impacted entries, six at depth1. They are `benchmark_microbatch.py`, `benchmark_workspace.py` (file and `main`), `microbatch_attention.py`, `test_openfold3_bias_layouts.py`, and `test_openfold3_workspace.py`. Depth2 adds `test_openfold3_microbatch.py`.
- [inferred] Keep the measured fixed wrappers intact and add a model-specific owner. This preserves reproduction callers and narrows the change. Dynamic hook installation and CPython exports still need explicit boundary tests despite the graph's exact local count.

## 5. Statement-Level PDG Findings

[graph] Bounded `pdg_query(controls, workspace_attention.py, limit=200)` returns114 edges. The relevant slice is `__call__`63–104 and `close`106–115. `flows(variable=self._scores)` returns zero edges; attribute data flow is not established by that result.

[verified] Source ordering is: lock64; guards68–79; first allocation80–85; mark submission86; HIP87–90; status guard91; PV93; private-storage guard94; successful-call counter97; poison on submission exception100–101; unlock104. Cleanup guards107, synchronizes111, then drops112. The plan preserves this ordering, including native PV before any same-stream overwrite. Extend model-boundary cleanup to interruptions as well as ordinary exceptions. Do not infer Python exception semantics solely from graph branch labels.

## 6. Proposed Changes

All names marked NEW below are proposed, not existing APIs.

1. [inferred] Add **NEW `WorkspaceLease`** in `crates/fireprot-core/src/workspace.rs`, exported by core. Own checked byte budgeting, a single lease, owner/device/stream identity, reservation/allocated/busy/failed/released state and receipts. A reservation may roll back before allocation commitment. Successful submission returns to reusable state only through explicit completion of host submission; this is distinct from GPU drainage. A failed owner cannot be reused.
2. [inferred] Add an isolated bridge module and additive export at `_fireprot_bridge`. Reject invalid state transitions and reentrant borrowing. Preserve registry APIs and ABI contracts. Rust receives metadata acknowledgments; it cannot prove GPU completion itself.
3. [inferred] Add **NEW `ModelWorkspaceFusion`** in `experiments/mfma_qk/model_workspace.py`. It owns exactly one B256 `WorkspaceFusion` plus one Rust lease. Direct B256 uses that pool; B512/B1024 iterate256 and concatenate independent native PV results. Normalize bias once per complete call. Reuse measured checks; do not create fixed `MicrobatchFusion` objects per shape. Check current HIP device as well as tensor device/stream, and prohibit graph capture.
4. [inferred] Add an explicit experimental worker overlay. Register only measured N1536/H4/D32/B256,512,1024 capabilities. Leave D16, N1000, B64 and unmatched layouts on the accepted dispatcher. Cached predicates describe immutable eligibility; dynamic lease checks run for every invocation. Allocation OOM or postsubmission errors fail the trial with diagnostics, without silent recovery that changes the treatment.
5. [inferred] Close in forward `finally`: drain, drop tensor, then release the Rust lease. Preserve original errors if cleanup also fails. Never report zero retained bytes after failed drain. Record allocation count/bytes, complete-call and kernel-launch counts, fallback reasons, device/stream identity, closure, Torch peak memory, tuner choices and existing cache counters. Distinguish Torch allocator reservation from live tensor bytes.

## 7. Implementation Sequence

1. Add Rust state policy and transition/failure tests; no model behavior changes.
2. Add the bridge export and cross-language tests; build and pin the new bridge/CLI.
3. Add the unified experimental tensor owner and deterministic CPU failure-path tests. Keep existing benchmark wrappers reproducible.
4. Qualify mixed B256/512/1024 calls on HIP, including head-last bias, broadcast/per-outer mask, fully masked rows, old-output independence and poisoned-workspace overwrite. Require bitwise scores/PV and one allocation.
5. Stage a new opt-in overlay from the frozen accepted D16 deployment after checking destination and source receipts. Run one N1536 model gate and N1000 fallback regression. Preserve FP32, strict fill, R3/four trunk passes, D200, S1 and seed101. Check all11 arrays and CIF against the accepted baseline and retained native anchors; investigate any failure before timing.
6. If model gates pass, run isolated three-pair AB/BA/AB E2E against the accepted D16 baseline. Both arms use the same newly pinned framework artifacts. Record cold/cache state, tuning and memory. No concurrent heavy build or GPU benchmark. Check known process handles before restarting anything after a network interruption.
7. Publish receipts and final provenance once, then decide whether to promote. An unchanged or slower result remains an experimental outcome. A new overall speedup versus original Torch requires a fresh direct comparison, not multiplication of earlier ratios.

## 8. Test Strategy

The context pack below specifies new state and tensor-boundary scenarios. Existing `tests/operators/test_openfold3_workspace.py`, `test_openfold3_microbatch.py`, `test_openfold3_bias_layouts.py` and `test_rust_registry.py` are regression gates. Build the bridge before importing the shared operator API. The verified Cargo workspace supports:

```bash
cargo test --workspace --locked
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
python -m pytest -q tests/operators/test_rust_registry.py tests/operators/test_openfold3_workspace.py tests/operators/test_openfold3_microbatch.py tests/operators/test_openfold3_bias_layouts.py
```

Run new test files when added. GPU commands must be generated from pinned deployment manifests; no invented runnable benchmark command is provided here. Gate model output before performance trials, and repeat correctness for every timed model run.

## 9. Risk and Impact Analysis

- [inferred] Retained9GiB can change autotuner choices or exhaust model memory even though isolated operator tests fit. Measure model peak; do not force upstream chunks merely to obtain a favorable result.
- [inferred] Closing after asynchronous work or a failed synchronization is the main lifecycle risk. Rust state cannot replace GPU synchronization at the tensor boundary.
- [inferred] Every graph depth1 dependent remains supported through unchanged existing wrapper signatures. If implementation requires changing a measured wrapper, rerun its direct benchmark and both CPU test families; graph count alone does not waive that gate.
- [verified] Registry capability results are cached; putting resource acquisition there would bypass dynamic checks on later calls.
- [inferred] The frozen accepted experimental worker must be preserved. Rebuilding only the candidate framework would confound the timing comparison.

## 10. Files Expected to Change

| File | Symbols | Reason |
| --- | --- | --- |
| `crates/fireprot-core/src/workspace.rs` | NEW WorkspaceLease | Typed single-owner budget and lifecycle policy; no GPU or tensor dependency. |
| `crates/fireprot-core/src/lib.rs` | module exports | Export the new policy module. |
| `crates/fireprot-bridge/src/workspace.rs` | NEW WorkspaceLease bridge | Expose narrow state operations and receipts through PyO3. |
| `crates/fireprot-bridge/src/lib.rs` | _fireprot_bridge | Register the additive class without changing existing registry APIs. |
| `experiments/mfma_qk/model_workspace.py` | NEW ModelWorkspaceFusion | One B256 tensor pool and one Rust lease shared across outer B256/512/1024 calls. |
| `experiments/mfma_qk/model-workspace.patch` | NEW experimental worker overlay | Install opt-in dispatch and finally cleanup on the frozen accepted D16 worker. |
| `experiments/mfma_qk/WORKSPACE.md` | Documentation | Record model acceptance, memory and timing scope after measurements. |

New test files are listed in §11. Any model-derived overlay retains its upstream Apache-2.0 notice; original resource policy follows the repository license.

## 11. Reusable Implementation Context

```json
{
  "implementation_context": {
    "task_summary": "Opt-in OpenFold3 model integration of the measured HIP QK/softmax kernel with one Rust-governed 9 GiB score workspace.",
    "acceptance_criteria": [
      "One shared B256 pool for B256/512/1024; no default promotion",
      "Strict FP32 R3/D200/S1 seed101, unchanged uninitialized-memory fill",
      "All11 arrays bitwise and CIF identity at N1536; N1000 fallback regression",
      "Truthful cleanup and allocation receipts",
      "Isolated AB/BA/AB paired E2E against accepted D16 baseline"
    ],
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "34e674ab4b82da83e35e08388ea9782b1c892973",
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
          "head_digest": "sha256:39b7152fb65901eed7ef459c98afa3f705880daf078538aaa0d00d31c89b9220",
          "index_digest": "sha256:39b7152fb65901eed7ef459c98afa3f705880daf078538aaa0d00d31c89b9220",
          "worktree_digest": "sha256:39b7152fb65901eed7ef459c98afa3f705880daf078538aaa0d00d31c89b9220",
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
          "head_digest": "sha256:b275714355e9cd32db12b779e1c942166fc5ddab4a797a1bad7c142625e1f084",
          "index_digest": "sha256:b275714355e9cd32db12b779e1c942166fc5ddab4a797a1bad7c142625e1f084",
          "worktree_digest": "sha256:b275714355e9cd32db12b779e1c942166fc5ddab4a797a1bad7c142625e1f084",
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
        "symbol": "WorkspaceFusion.__call__",
        "file": "experiments/mfma_qk/workspace_attention.py",
        "lines": "63-104",
        "role": "Measured tensor allocation, launch and native PV ordering"
      },
      {
        "symbol": "WorkspaceFusion.close",
        "file": "experiments/mfma_qk/workspace_attention.py",
        "lines": "106-115",
        "role": "Owner guard, GPU drain and tensor cleanup"
      },
      {
        "symbol": "_fireprot_bridge",
        "file": "crates/fireprot-bridge/src/lib.rs",
        "lines": "148-156",
        "role": "Additive Rust policy export"
      },
      {
        "symbol": "AttentionDispatcher",
        "file": "python/fireprot/adapters/openfold3/attention.py",
        "lines": "87-165",
        "role": "Existing Rust registry selection seam"
      },
      {
        "symbol": "forward_with_backend",
        "file": "python/fireprot_workers/openfold3.py",
        "lines": "116-145",
        "role": "Forward scope and manifest boundary"
      }
    ],
    "related_symbols": [
      {
        "symbol": "MicrobatchFusion",
        "relationship": "IMPORTS WorkspaceFusion",
        "relevance": "Current fixed outer-batch wrappers each own a pool; do not instantiate one per model batch"
      },
      {
        "symbol": "OperationRegistry",
        "relationship": "dispatch seam",
        "relevance": "Rust selection caches capability results; dynamic lease state belongs at invocation"
      }
    ],
    "execution_path": [
      "Enter experimental forward scope",
      "Register immutable shape capability with existing registry",
      "Eligible invocation validates current owner/device/stream",
      "Reserve lease and lazily allocate B256 scores once with ordinary strict fill",
      "Normalize bias once; run B256 kernel and native PV for each microchunk",
      "Return independent output; retain private scores until forward finally",
      "Synchronize, drop tensor, then acknowledge released lease and write receipts"
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
        "file": "crates/fireprot-core/src/workspace.rs",
        "symbols": [
          "NEW WorkspaceLease"
        ],
        "intended_change": "Typed single-owner budget and lifecycle policy; no GPU or tensor dependency."
      },
      {
        "file": "crates/fireprot-core/src/lib.rs",
        "symbols": [
          "module exports"
        ],
        "intended_change": "Export the new policy module."
      },
      {
        "file": "crates/fireprot-bridge/src/workspace.rs",
        "symbols": [
          "NEW WorkspaceLease bridge"
        ],
        "intended_change": "Expose narrow state operations and receipts through PyO3."
      },
      {
        "file": "crates/fireprot-bridge/src/lib.rs",
        "symbols": [
          "_fireprot_bridge"
        ],
        "intended_change": "Register the additive class without changing existing registry APIs."
      },
      {
        "file": "experiments/mfma_qk/model_workspace.py",
        "symbols": [
          "NEW ModelWorkspaceFusion"
        ],
        "intended_change": "One B256 tensor pool and one Rust lease shared across outer B256/512/1024 calls."
      },
      {
        "file": "experiments/mfma_qk/model-workspace.patch",
        "symbols": [
          "NEW experimental worker overlay"
        ],
        "intended_change": "Install opt-in dispatch and finally cleanup on the frozen accepted D16 worker."
      },
      {
        "file": "experiments/mfma_qk/WORKSPACE.md",
        "symbols": [],
        "intended_change": "Record model acceptance, memory and timing scope after measurements."
      }
    ],
    "tests": [
      {
        "file": "crates/fireprot-core/tests/workspace.rs (new)",
        "scenarios": [
          "Insufficient budget or overflow -> reject before reservation",
          "Reserve twice or wrong owner/device/stream -> reject without state mutation",
          "Allocation failure -> rollback reservation; committed allocation -> prohibit rollback",
          "Submit or cleanup failure -> retain failed state/accounting; successful drain and tensor drop -> release exactly once",
          "Closed lease -> reject later use; mixed outer batches -> one reservation"
        ]
      },
      {
        "file": "tests/operators/test_openfold3_model_workspace.py (new)",
        "scenarios": [
          "B256, B512, B1024 sequence -> one pool and one allocation",
          "D16/N1000/B64/unmeasured layout -> existing accepted fallback without allocation",
          "GPU launch or PV exception -> poison and propagate; never call fallback",
          "Cleanup sync failure -> no successful release receipt",
          "Allocation exception and asynchronous interruption -> cleanup preserves original error and truthful accounting",
          "Registry cache hit -> invocation still checks dynamic owner/stream/lifetime",
          "Same-stream calls -> previous outputs remain independent; different stream/thread/capture -> reject"
        ]
      },
      {
        "file": "tests/operators/test_rust_registry.py",
        "scenarios": [
          "Existing registry caching, concurrency and reentrancy semantics remain unchanged with additive bridge export"
        ]
      },
      {
        "file": "tests/operators/test_openfold3_workspace.py",
        "scenarios": [
          "Existing budget, strict setting, owner and close guards remain passing"
        ]
      },
      {
        "file": "tests/operators/test_openfold3_microbatch.py",
        "scenarios": [
          "Existing fixed-wrapper metadata and concurrency guards remain passing"
        ]
      },
      {
        "file": "tests/operators/test_openfold3_bias_layouts.py",
        "scenarios": [
          "Measured dense layouts remain accepted only with required normalization"
        ]
      }
    ],
    "verification_commands": [
      "cargo test --workspace --locked",
      "cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked",
      "python -m pytest -q tests/operators/test_rust_registry.py tests/operators/test_openfold3_workspace.py tests/operators/test_openfold3_microbatch.py tests/operators/test_openfold3_bias_layouts.py"
    ],
    "risks": [
      "Retained 9 GiB can alter model chunk tuning or cause allocation failure",
      "Python callback and tensor ownership cross a static graph boundary",
      "Sync failure leaves resource state unresolved until process teardown",
      "New bridge/CLI artifacts must be rebuilt and pinned for both benchmark arms"
    ],
    "assumptions": [
      "The frozen accepted D16 deployment is still available: verify its source and artifact receipts before staging the overlay.",
      "The measured combined-v12 bundle is unchanged: verify its manifest before loading."
    ],
    "open_questions": [
      "Actual N1536 model peak memory and autotuner choices with retained workspace",
      "Mixed-shape full-model bitwise compatibility and E2E effect"
    ],
    "avoid": [
      "Do not repeat full repository discovery",
      "Do not create separate 9 GiB pools for each outer batch",
      "Do not disable strict deterministic execution or uninitialized-memory filling",
      "Do not change HIP arithmetic, reduction order, precision, recycle count or diffusion steps",
      "Do not fall back after partial GPU submission",
      "Do not multiply incremental speedup ratios into a direct original-Torch claim",
      "Do not commit docs/isa/CDNA2.pdf"
    ]
  }
}
```

## 12. Assumptions and Open Questions

- [assumed] Frozen D16 deployment and combined-v12 bundle remain available; check existing source/artifact receipts before reuse. Expected kernel manifest: `1f04b110c79520749dc2a9a483b77f35330b559f5c043dfcdd13c72947ec5122`.
- [assumed] Model memory permits the pool; resolve by a measured N1536 preflight and full run, not by extrapolating operator peaks.
- Open: mixed-shape model correctness, tuner changes and actual E2E benefit. No estimate is an accepted result.
- Source-weighted limitation: MCP impact's stale label disagrees with matching context and typed CLI status. PDG does not establish tensor attribute data flow or dynamic callback coverage.
- Deferred: B64 optimization, other sequence lengths, new arithmetic/PV fusion, generalized cross-model workspace services and OpenFold2 work.

## 13. Definition of Done

- One bounded shared workspace; normal and exceptional forward exits have truthful resource receipts.
- Rust owns policy; Python/Torch owns allocation and GPU execution; existing fallback and wrapper behavior remain tested.
- Full N1536 and N1000 outputs pass all11 bitwise arrays and CIF identity under the fixed scientific contract.
- Isolated paired timing, memory and tuning evidence is archived with exact source/binary/input/model identities.
- Documentation states the measured comparison scope. Promotion requires demonstrated E2E benefit; otherwise the result stays opt-in and the limiting evidence is recorded.
