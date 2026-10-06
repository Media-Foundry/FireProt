# GitNexus Engineering Plan

> Task: Complete one additional attention optimization cycle, then improve README.
> Evidence verified at commit 9aab0cd3c5dbda892be1141dacdc231970e2d6c9; current schema-4 analyzer, index refreshed --index-only --pdg.
> Evidence provenance schema 2, canonical dirty digest and sorted manifest in section 11.

## 1. Objective

Follow the user's sequence: complete the active triangle-memory cycle; perform one additional measured optimization cycle; then finish the English README with clear usage and honest E2E comparisons.

## 2. Current Behaviour

[verified] Bounded actual N1536 attention profiling records approximately144ms QK+softmax and71ms PV per call. The first cycle improves multiplication copies substantially and key packing modestly. Its combined full-model all11/fullCIF gate passes; guarded N1000 and three-pair E2E remain active under controller2235527.
[verified] The accepted attention workspace retains one ordinarily allocated4.5GiB score tensor, preserving strict uninitialized-memory filling. Current QK+softmax uses combined-v12, native PV and independent output storage.

## 3. Relevant Architecture

[verified] `WorkspaceFusion.__call__` checks strict settings, owner, lock, device/stream and graph-capture restrictions before writing private scores; failures after submission close it. `ModelWorkspaceFusion` delegates bounded128-row slices under a Rust lease. Rust owns build identity and model/paired supervision. Python remains tensor glue; HIP implements GPU kernels.

## 4. GitNexus Findings

[graph] context(WorkspaceFusion,workspace_attention.py) identifies model workspace, microbatch and benchmark imports and the workspace/bias-layout tests. Existing consumers are not edited in the initial experiment. New symbols have no indexed consumers; verify experiment/model-scoped dynamic patching manually. Typed context and CLI prove freshness; stale MCP prose counters are ignored.

## 5. Statement-Level PDG Findings

[graph] Bounded controls query returns10 of114 edges: owner/lock failure at107 gates raising at108; device presence110 gates synchronization111 before release112. This is a partial slice, not whole-function completeness. [verified] Source confirms acquire/check/submit/nativePV/close/finally-release sequence; retain it for both alternatives. No inferred PDG edges are claimed for new HIP kernels.

## 6. Proposed Changes

- NEW isolated `experiments/attention_final/` alternative A: native `torch.bmm` into the retained contiguous score buffer, followed by existing owned-score HIP mask/bias/softmax and unchanged native PV. Retain the first cycle's winning K packing. This separates handwritten QK fusion from score allocation reuse; verify bmm(out=) output bits and selected kernel, not just tensor shapes.
- Alternative B: reconstruct combined-v12 from its committed base and documented six-patch chain, verify reconstructed hash, then alternate two16KiB LDS buffers. Retain six producer barriers; remove only five consumer/reuse barriers. Preserve every MFMA, operand order, mask/bias addition and softmax reduction operation. Build/disassemble: prove32KiB LDS, addressing in range, expected barriers and no spills. Resource counts alone cannot establish performance.
- Screen both alternatives and run full trained start/end attention replay. Only an exact, repeatably faster complete operator proceeds to model gates. Retain rejected candidates. A loss is a valid outcome for this final cycle.
- After cycle2 completes: update README with real Cargo build and CLI examples, pinned deployment prerequisites, model compatibility, maintained versus experimental entry points, and E2E/model timing tables with workload/cache/isolation limits. Recheck actual commands and source before publication.

## 7. Implementation Sequence

1. Observe original controller2235527 to terminal; preserve failed receipts, diagnose failures. No new GPU work or remote heavy build while timed work runs.
2. Create isolated alternatives and numerical/lifecycle screens; Rust builds only after prior gate. Apply source pins and explicit mode provenance.
3. Bitwise QK/probability/PV screen (existing adversarial cases), then actual trained attention balanced replay. Keep allocation and K-packing equal across arms.
4. Winning arm: frozen N1536 R3/D200/S1 seed101 full all11/fullCIF against native,D16,latest and cycle1; N1000 fallback. No winning operator means retain cycle1 and document rejection.
5. Three-pair isolated incremental E2E under one stable inventory; validate every output. Obtain direct original-Torch paired evidence for final accepted combination when feasible; until then use the existing direct ratio with explicit version/scope, never products of incremental ratios.
6. Review docs and actual entry commands; finish README and linked reproduction docs in English, retain original Chinese baseline. Commit/push sources, evidence and documentation. Stop further optimization after this additional cycle as requested.

## 8. Test Strategy

Existing `tests/operators/test_openfold3_workspace.py` supplies owner, lock, closed, budget and strict-mode rejection patterns. New screens preserve those contracts. Compare random and adversarial finite logits, masking, NaN behavior where applicable, raw scores, probabilities, final PV bits; include actual trained inputs. Double-buffer stress repeatedly reuses retained output and nondefault stream. Include actual K packing, allocation and PV in timing. First calls excluded and balanced repetitions; model and E2E gates remain independent. Syntax command: `python -m compileall -q experiments/attention_final` after creation. Rust paired commands are the same validated cycle1 runner.

## 9. Risk and Impact Analysis

[inferred] bmm(out=) may change library dispatch, hidden workspace or numerical order; require bitwise tests and profiling. Double LDS can alter occupancy; no speed promise. All listed WorkspaceFusion consumers keep their original implementation outside explicit experimental scopes. Dynamic method replacement requires owner isolation and finally restoration. Published runtime manifests must identify split versus fused execution accurately, including source/library pins and counts. No mutation of active frozen workers.

## 10. Files Expected to Change

| Path | Responsibility |
| --- | --- |
| experiments/attention_final/ | Isolated candidates, screens, replay and scope |
| provenance/openfold3_attention_final_* | Actual acceptance/performance and rejection evidence |
| README.md | Clear English entry points and measured comparison table |
| docs/openfold3-performance.md | Detailed accepted scope and reproduction |
| .agents/project_context.md | Append-only continuation state |

## 11. Reusable Implementation Context

```json
{
  "implementation_context": {
    "task_summary": "One final OpenFold3 attention optimization cycle, then complete English README",
    "acceptance_criteria": [
      "Current triangle-memory E2E terminal before next GPU work",
      "Exact probabilities, PV and real trained attention",
      "Full model all11/fullCIF and N1000 fallback",
      "Paired E2E before promotion",
      "README documents actual entry points and defensible speedup scopes"
    ],
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "9aab0cd3c5dbda892be1141dacdc231970e2d6c9",
      "generated_plan_path": "docs/plans/2026-10-06-gitnexus-plan-openfold3-attention-final-cycle.md",
      "global_dirty_digest": {
        "algorithm": "sha256",
        "canonicalization": "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records",
        "value": "af566e7d76722cf710abc0aa507f71abf4935108f52433779eba62763ee3725d"
      },
      "cited_path_manifest": [
        {
          "path": "experiments/hip_softmax/owned_scores.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:3ff67c26648a2f3a6de8dc486191e567cc9831f441a6d02161e53eddbcd7e390",
          "index_digest": "sha256:3ff67c26648a2f3a6de8dc486191e567cc9831f441a6d02161e53eddbcd7e390",
          "worktree_digest": "sha256:3ff67c26648a2f3a6de8dc486191e567cc9831f441a6d02161e53eddbcd7e390",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/FUSION.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:d7559a02a73ddf673233b70686c53f9c774826d325d181c95a0cdbf949096f0d",
          "index_digest": "sha256:d7559a02a73ddf673233b70686c53f9c774826d325d181c95a0cdbf949096f0d",
          "worktree_digest": "sha256:d7559a02a73ddf673233b70686c53f9c774826d325d181c95a0cdbf949096f0d",
          "untracked_digest": "absent"
        },
        {
          "path": "experiments/mfma_qk/fused_qk_softmax.cpp",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:9605b069c16fb53abff832991c6dc6eeb1018066d0c854c38b463fd2eae09a97",
          "index_digest": "sha256:9605b069c16fb53abff832991c6dc6eeb1018066d0c854c38b463fd2eae09a97",
          "worktree_digest": "sha256:9605b069c16fb53abff832991c6dc6eeb1018066d0c854c38b463fd2eae09a97",
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
          "head_digest": "sha256:47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d",
          "index_digest": "sha256:47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d",
          "worktree_digest": "sha256:47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d",
          "untracked_digest": "absent"
        },
        {
          "path": "provenance/openfold3_triangle_triatt_profile.json",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:7944f8bc31de78fed9b4680517043bdd7443ff8362af2809118bd6b99045b4a1",
          "index_digest": "sha256:7944f8bc31de78fed9b4680517043bdd7443ff8362af2809118bd6b99045b4a1",
          "worktree_digest": "sha256:7944f8bc31de78fed9b4680517043bdd7443ff8362af2809118bd6b99045b4a1",
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
        }
      ]
    },
    "primary_symbols": [
      {
        "symbol": "WorkspaceFusion.__call__",
        "file": "experiments/mfma_qk/workspace_attention.py",
        "lines": "65-104",
        "role": "Existing checked workspace lifecycle"
      },
      {
        "symbol": "OwnedScoreSoftmax.__call__",
        "file": "experiments/hip_softmax/owned_scores.py",
        "lines": "32-61",
        "role": "Existing exact in-place softmax"
      }
    ],
    "related_symbols": [],
    "execution_path": [
      "Existing strict qualified Q/K packing",
      "One retained score allocation",
      "Candidate QK and exact softmax",
      "Unchanged native PV",
      "Independent returned output"
    ],
    "pdg_constraints": [
      {
        "description": "Workspace owner/lock guard precedes drain; device synchronization precedes score release",
        "affected_statements": [
          "experiments/mfma_qk/workspace_attention.py:107",
          "experiments/mfma_qk/workspace_attention.py:111"
        ],
        "implementation_consequence": "Retain lifecycle guards and fail-closed launch behavior"
      }
    ],
    "architectural_patterns": [
      {
        "pattern": "Explicit experimental scope and Rust-owned build/benchmark",
        "example_location": "experiments/mfma_qk/workspace_attention.py",
        "usage_guidance": "New experiment modules; retain existing fallback"
      }
    ],
    "files_to_modify": [
      {
        "file": "experiments/attention_final/",
        "symbols": [
          "NEW split workspace",
          "NEW double-buffer candidate",
          "NEW screen/replay"
        ],
        "intended_change": "Compare two bounded alternatives"
      },
      {
        "file": "README.md",
        "symbols": [],
        "intended_change": "After second cycle, document build, maintained and experimental usage, support and E2E table"
      }
    ],
    "tests": [
      {
        "file": "experiments/attention_final/",
        "scenarios": [
          "Finite/adversarial/masked QK, probabilities and PV exact",
          "Actual trained start/end attention exact",
          "Owner,stream,alias,closed checks retained",
          "All11/fullCIF N1536 and N1000 native fallback"
        ]
      }
    ],
    "verification_commands": [
      "python -m compileall -q experiments/attention_final"
    ],
    "risks": [
      "bmm(out=) may select another kernel or hidden allocation",
      "Double LDS may change occupancy or resource use",
      "No benefit is a valid final-cycle outcome"
    ],
    "assumptions": [
      "Double-buffer producer barrier orders previous consumer before buffer reuse; verify generated ISA and GPU stress",
      "Current cycle yields an accepted baseline; if it fails, diagnose before advancing"
    ],
    "open_questions": [
      "Whether native bmm out plus HIP softmax beats fused QK",
      "Whether removing five reuse barriers offsets doubled LDS",
      "Current complete E2E ratios pending"
    ],
    "avoid": [
      "No overlapping GPU work or heavy build during paired benchmarks",
      "No scientific workload or precision relaxation",
      "No Triton",
      "No multiplication of incremental ratios",
      "Do not commit user ISA PDF",
      "Do not continue unbounded optimization after this additional cycle"
    ]
  }
}
```

## 12. Assumptions and Open Questions

[assumed] Alternating buffers plus the intervening producer barrier prevents overwrite-before-read; verify address arithmetic and generated ISA, then GPU stress. [assumed] Current cycle supplies an accepted baseline; a failed gate must be resolved first. Both speedups are unknown. This is one bounded additional cycle, not permission for endless tuning. Direct original-Torch timing may be unavailable until a fresh comparison completes; documentation must state that limitation.

## 13. Definition of Done

Current cycle resolved; one additional cycle measured and accepted or rejected with exact evidence; no failed candidate promoted. README gives runnable entry points and correctly scoped E2E/model ratios with model/workload/cache/isolation details. Code, evidence and docs committed/pushed; ISA remains untracked. No unmeasured performance claim.
