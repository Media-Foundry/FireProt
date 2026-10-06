# GitNexus Engineering Plan

> Task: Optimize OpenFold3 triangle multiplication and attention memory traffic.
> Evidence verified at commit 504dc14f99be73c8e4d0dcc89d0fe4b78c9bf97a; index refreshed with --index-only --pdg, current schema-4 runner.
> Evidence provenance schema 2; full manifest and dirty digest appear in section 11.

## 1. Objective

Reduce long-protein latency while preserving strict FP32 R3/D200/S1 seed101 semantics and bitwise public outputs.

## 2. Current Behaviour

[verified] The retained fine profile records 110.790 seconds in trunk triangle attention and 97.205 seconds in triangle multiplication (`provenance/openfold3_latest_pairformer_profile.json`). These are diagnostic, nested GPU intervals, not E2E measurements.
[verified] Completed remote multiplication capture uses standard incoming/outgoing modules, FP32 [1,1536,1536,128], chunk256 and in-place residual. Its all11/fullCIF checks pass against native, D16 and latest workspace anchors. Projection packing and implicit BMM/LN copies consume substantial time. Raw evidence is retained at `/tmp/fireprot-trimul-profile-v1` on DiamondHill; do not discard it.

## 3. Relevant Architecture

[verified] `python/fireprot/backends/amd/_native_hip.py:12-56` assembles source/tool/header fingerprints and delegates builds to Rust before loading one ctypes artifact. Reuse it unchanged. Python remains tensor glue; HIP implements GPU copying; Rust supervises inference and acceptance.

## 4. GitNexus Findings

[graph] context(load,file_path=python/fireprot/backends/amd/_native_hip.py) reports six direct callers including experiment builders and CK/BLAS loaders. None are edited. New experimental symbols have no existing indexed consumers. Upstream OpenFold3 lives outside this index and requires source pinning/manual call tracing.
[verified] Typed context and CLI status agree on current HEAD and runner; context tool's prose behind-count is inconsistent and is not used as freshness evidence.

## 5. Statement-Level PDG Findings

[graph] pdg_query(controls,target=python/fireprot/backends/amd/_native_hip.py) returns zero edges; this is not proof of absent guards. Source verifies missing ldd and unexpected artifact-count errors. Preserve those errors. Proposed kernels are NEW and have no indexed PDG. Preserve source-before-destination copy ordering, explicit stream ordering and native residual aliasing.

## 6. Proposed Changes

- NEW `experiments/triangle_memory/copy.cpp`: tiled uint32 strided transpose, padded LDS tile, explicit stream C ABI. No floating arithmetic.
- NEW `experiments/triangle_memory/copy_runtime.py`: qualify only measured FP32 layouts, non-overlapping storage and supported dimensions; native prelaunch fallback; propagate launch errors.
- NEW experiment replay and screen scripts: pinned native method extraction with exact substitutions limited to projection assignments, BMM input packing and output LayerNorm packing. Keep native GEMM shapes and arithmetic. Preserve Apache attribution for copied upstream code.
- Attention track: collect pending bounded profile and trained snapshots; rank self-device costs, implement only qualified exact layout/copy improvements under this plan. An arithmetic rewrite requires separate planning.
- Add English documentation, diagnostic provenance and append-only project journal records.

## 7. Implementation Sequence

1. Commit plan and collect completed bounded diagnostics.
2. Implement isolated HIP copy and meaningful bitwise/offset/stream/fallback screening.
3. Replay trained incoming/outgoing multiplication; measure complete operator in alternating order. Reject parity failures.
4. Use attention profile to implement and replay its independently qualified memory improvement; report if memory changes offer no gain.
5. Only winning candidates enter fresh frozen model workers, N1536 exact all11/CIF against all three anchors, N1000 fallback check.
6. Stable-inventory isolated three-pair E2E against latest accepted candidate, then direct original native comparison. Archive and promote only accepted gains; keep rejected trials.

## 8. Test Strategy

Use arbitrary uint32 bit patterns viewed as FP32, including NaNs and signed zeros; projection offsets 0/256/1280; untouched destination canaries; source immutability; default and nondefault streams; alias/unsupported fallback. Replay actual trained captures, not random-only projections. Python syntax command is `python -m compileall -q experiments/triangle_memory`; GPU scripts run with HIP_VISIBLE_DEVICES=7 and the pinned OpenFold3 interpreter after the active diagnostic exits. Full model acceptance remains Rust-owned.

## 9. Risk and Impact Analysis

[inferred] Explicit packing can select different GEMMs or change chunk tuning through memory pressure. Operator bits alone cannot prove model acceptance or E2E gains. Retain native fallback and record allocation. No maintained callers change in the isolated stage; six loader consumers remain unchanged. Do not overlap GPU work or mutate timed deployment. Preserve upstream Apache licenses.

## 10. Files Expected to Change

| Path | Responsibility |
| --- | --- |
| experiments/triangle_memory/ | New copy kernel, tensor glue, replay and screens |
| provenance/openfold3_triangle_* | Exact evidence and rejected trials |
| docs/openfold3-performance.md | Measured results only |
| .agents/project_context.md | Append-only continuation record |
| THIRD_PARTY.md | Any copied upstream attribution |

## 11. Reusable Implementation Context

```json
{
  "implementation_context": {
    "task_summary": "Optimize measured OpenFold3 triangle multiplication and attention memory costs",
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "504dc14f99be73c8e4d0dcc89d0fe4b78c9bf97a",
      "generated_plan_path": "docs/plans/2026-10-06-gitnexus-plan-openfold3-triangle-memory-optimization.md",
      "global_dirty_digest": {
        "algorithm": "sha256",
        "canonicalization": "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records",
        "value": "2c6402b0053bf1b1d85ccb785f269c24e0d3531ff1b0a288fad3cf5b9ec36643"
      },
      "cited_path_manifest": [
        {
          "path": "provenance/openfold3_latest_pairformer_profile.json",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:33d006f73fa954a3d05ca1fbff0653e5b936ce2b54164fe19b4d2ed8599acdf7",
          "index_digest": "sha256:33d006f73fa954a3d05ca1fbff0653e5b936ce2b54164fe19b4d2ed8599acdf7",
          "worktree_digest": "sha256:33d006f73fa954a3d05ca1fbff0653e5b936ce2b54164fe19b4d2ed8599acdf7",
          "untracked_digest": "absent"
        },
        {
          "path": "python/fireprot/backends/amd/_native_hip.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:60052815e80450203762c425a78d97d92583a9ae3297bc410a70d22aaed92dbd",
          "index_digest": "sha256:60052815e80450203762c425a78d97d92583a9ae3297bc410a70d22aaed92dbd",
          "worktree_digest": "sha256:60052815e80450203762c425a78d97d92583a9ae3297bc410a70d22aaed92dbd",
          "untracked_digest": "absent"
        }
      ]
    },
    "acceptance_criteria": [
      "Bitwise operator parity",
      "All 11 arrays and CIF bytes match three anchors",
      "Isolated paired E2E improvement before promotion"
    ],
    "primary_symbols": [
      {
        "symbol": "load",
        "file": "python/fireprot/backends/amd/_native_hip.py",
        "lines": "12-56",
        "role": "Reuse unchanged Rust build boundary"
      }
    ],
    "related_symbols": [],
    "execution_path": [
      "Pinned captured input",
      "Qualified HIP copy",
      "Unchanged native arithmetic",
      "Rust full-model acceptance"
    ],
    "pdg_constraints": [],
    "architectural_patterns": [
      {
        "pattern": "Rust native HIP build with explicit ctypes stream",
        "example_location": "python/fireprot/backends/amd/_native_hip.py:12",
        "usage_guidance": "Reuse without modification"
      }
    ],
    "files_to_modify": [
      {
        "file": "experiments/triangle_memory/",
        "symbols": [
          "NEW tiled_copy",
          "NEW CopyRuntime",
          "NEW screen"
        ],
        "intended_change": "Isolated experiments"
      }
    ],
    "tests": [
      {
        "file": "experiments/triangle_memory/screen.py",
        "scenarios": [
          "Bit patterns including NaN and signed zero remain identical",
          "Offsets and canaries remain valid",
          "Unsupported layouts and aliasing use native fallback",
          "Captured trained operators preserve all bits"
        ]
      }
    ],
    "verification_commands": [
      "python -m compileall -q experiments/triangle_memory"
    ],
    "risks": [
      "Changed allocation can affect tuning",
      "Explicit packing can change library selection"
    ],
    "assumptions": [
      "Measured copy shapes qualify; inspect captured input strides before replay"
    ],
    "open_questions": [
      "Attention operator profile pending",
      "E2E gain unknown"
    ],
    "avoid": [
      "No arithmetic or recycle change",
      "No concurrent GPU benchmark",
      "No Triton",
      "Do not commit ISA PDF",
      "Do not claim incremental ratios as direct original-Torch speedup"
    ]
  }
}
```

## 12. Assumptions and Open Questions

[assumed] Measured copy shapes qualify; confirm strides in captured replay before launch. Attention capture is running; do not guess its remaining bottleneck. E2E gain and changed peak allocation are unknown. New arithmetic, recycle cuts and template skipping are out of scope. The historical direct original-Torch result remains 1.499x; latest workspace 1.201x is incremental.

## 13. Definition of Done

Both target tracks have measured accepted improvements or documented exhausted memory candidates. Every candidate has exact operator checks; promotion requires full-model all11/CIF identity and stable three-pair E2E. Preserve failed trials and remaining limitations. No unsupported speedup claim.
