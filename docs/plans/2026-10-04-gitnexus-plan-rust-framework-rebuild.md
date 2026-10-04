# GitNexus Engineering Plan

> Task: rebuild the maintained FireProt framework in Rust and organize the repository before resuming OpenFold3 optimization.
> Evidence verified at c2dd7535aa652ff0e2e8d9a12d8e83709042edda; index current, schema4 runner current, no incomplete reasons. Autonomous continuation uses a full architecture plan.

## 1. Objective

Move framework ownership to Rust: requests, backend policy, job lifecycle, device
isolation, artifact provenance, numerical validation, benchmark summaries, and
maintained command-line workflows. Keep explicit upstream model-worker and
HIP/CK boundaries. Finish migration validation before another optimization cycle.
A Rust executable that merely forwards the old Python benchmark CLI is insufficient.

## 2. Current Behaviour

- [verified] `scripts/infer.sh` owns Protenix defaults and environment, then runs
  an upstream or FireProt Python entry point. AtlasFold and OpenFold3 use separate
  benchmark entry points with their own configuration and manifest formats.
- [verified] `fireprot/ops/registry.py` owns registration ordering, capability
  selection, cached decisions, counts, and native fallback. Candidate exceptions
  reject that implementation rather than selecting it.
- [verified] GPU process monitoring lives in `benchmarks/protenix/profile_inference.py:199`
  and is imported by the other benchmark harnesses. AtlasFold supplies the newer
  PCI-to-SMI device resolver. This is framework logic inside benchmark packages.
- [verified] `fireprot/validation/protenix.py` implements Kabsch alignment and
  numeric confidence comparison. Missing files, schema differences, and nonfinite
  values must not turn into successful comparisons.
- [verified] There is no root Cargo workspace. Runtime data and predictions are
  ignored; `Protenix/` is an independent ignored Git checkout.

## 3. Relevant Architecture

Use five responsibilities, introduced when each has substantive implementation:

| Location | Responsibility | Allowed dependencies |
| --- | --- | --- |
| `crates/fireprot-core` | Versioned requests/events, model identity, tensor/operation contracts, deterministic registry | Rust data/serialization utilities only |
| `crates/fireprot-runtime` | Worker processes, device mapping/leases, monitoring, hashes/build identity, manifests | core; host OS and HIP/SMI probes |
| `crates/fireprot-validation` | Array/CIF/confidence comparison and repeat statistics | core; numerical/file-format crates |
| `crates/fireprot-cli` | User commands, diagnostics, exit codes | public interfaces of other crates |
| `crates/fireprot-bridge` | Small C ABI for Python model glue to call Rust policy | core; no model weights or tensor ownership |
| `python/fireprot_workers` | Three upstream model workers and tensor-specific hooks | pinned upstream model environments; bridge |
| `kernels/amd` | HIP/CK sources and build descriptors | ROCm, CK, native Torch bindings where required |
| `configs`, `tests/fixtures`, `docs` | Explicit profiles, offline fixtures, concise maintained documentation | no runtime-generated artifacts |

[inferred] A process boundary isolates incompatible model dependencies without
introducing a new tensor transport into GPU hot paths. Tensor execution stays in
the model process; Rust controls the job and policy. The small in-process bridge
lets operator selection migrate without copying device tensors through JSON.

## 4. GitNexus Findings

- [graph] `query` located registry, benchmark monitors and validation flows.
  Clusters include Operators, AMD, Csrc, Atlasfold and Validation.
- [graph] `impact(resolve, upstream, depth3)` reports `OperationRegistry.dispatch`
  directly, triangle attention/multiplication transitively. Receiver gaps make the
  three reported consumers a lower bound, not a complete inventory.
- [verified] Source search additionally finds registration in CK attention/GEMM,
  recycle and triangle modules; these must be included in the migration inventory.
- [verified] `tests/operators/test_registry.py` covers priority, cache invalidation,
  native fallback and predicate exceptions. `tests/benchmarks/test_protenix_profile.py`
  includes missing-SMI failure and PID selection checks.

## 5. Statement-Level PDG Findings

- [graph] The registry query at line150 found no statement block; use source
  evidence for its exception-to-rejection branch, not an invented PDG edge.
- [graph] The monitor slice at line227 reaches command execution and the
  successful-result guard; the depth1 slice is truncated and not exhaustive.
- [verified] Source confirms nonzero sample count, no command errors, and no
  unexpected PIDs are all required for an isolation pass. Rust must reject
  malformed/missing device records rather than treat them as an empty GPU.
- [verified] Registry ordering is descending priority followed by name/version;
  changes clear cached decisions. Rust must preserve deterministic tie handling
  and must not reuse a decision after its registry generation changes.

## 6. Proposed Changes

New Rust types/modules implement these responsibilities; names are proposed,
not claims about existing symbols:

- Core: schema version, tagged model request, explicit numerical workload,
  backend intent, artifact reference, tensor contract, worker events and errors.
  Recycle count remains model-specific; Protenix cycles must not be interpreted
  as AtlasFold/OpenFold3 additional recycles. Unknown schemas and invalid counts
  are rejected. No implicit reduction of inference work.
- Registry: Rust owns candidate ordering, policy decisions, rejections and counts.
  Model-specific tensor inspection may remain in workers, but shared selection
  policy must not remain duplicated in Python. A bridge handles local callbacks
  or serialized descriptors; do not call a subprocess per operator.
- Runtime: validate requests, pin source/weights, reserve a new output directory,
  spawn a worker without a shell, receive a device-ready event, independently
  match PCI/SMI, start continuous monitoring, then authorize execution. Preserve
  failure reports on exit, timeout, cancellation, malformed output or missing
  predictions. The worker can never declare its own isolation pass authoritative.
- Worker protocol: versioned JSON frames, reserved protocol channel separated
  from upstream logs, explicit ready/start/completed/failed states and run IDs.
  Small control data only; coordinate/confidence arrays are immutable artifacts
  with paths, shapes, original dtypes and hashes verified by Rust.
- Validation: exact bytes/shape/dtype by default; model-scoped predeclared tolerance
  policies for established Protenix comparisons, rigid alignment and confidence
  schema checks. No tolerance derived after seeing candidate errors.
- Layout: move maintained glue under `python/`, kernels under `kernels/amd`, and
  expose a root Cargo workflow. Keep old paths as compatibility shims only when
  needed by upstream patches; list every surviving Python responsibility.
  Historical scripts/evidence retain source identities or an explicit path map.
- Documentation: concise README with quickstart and supported-model matrix;
  detailed historical measurements stay in their existing performance reports.
  Replace stale architecture/context summaries after runtime validation.

## 7. Implementation Sequence

1. Commit this plan. Establish Cargo workspace, typed core request/protocol,
   semantic validation and offline tests. Add CLI request inspection once usable.
2. Implement Rust registry and comparison/provenance primitives. Replay existing
   fixtures, test invalid masks/layouts/architecture and cache generation changes.
3. Implement Rust device/process supervision and versioned worker protocol using
   fake workers. Prove malformed/stale/missing outputs and monitor errors fail.
4. Extract native model execution into explicit Protenix/AtlasFold/OpenFold3
   workers. Eliminate shared framework policy from these workers. Integrate
   Rust registry bridge with existing capability gates and HIP/CK entry points.
5. Migrate backend build identity/cache management and repository layout using
   checked path maps; preserve historical scripts and independent model checkouts.
6. Run full model parity on DiamondHill, including existing long workloads and
   native fallback. Compare supported accelerated routes with saved references;
   source-path/build changes require fresh GPU evidence. Record timing scope and
   migration overhead without promising an optimization.
7. Switch maintained entry points to Rust, update README/architecture/context,
   CI and examples, archive compatibility status, commit/push final evidence.
   Resume OpenFold3 tuning only after these gates pass.

## 8. Test Strategy

New Rust tests must discriminate failures rather than merely mirror fields:

- malformed/unknown protocol, invalid model-specific workloads, zero samples,
  forbidden dtype/model pair, unsupported backend, and missing checkpoint role;
- deterministic selection ties, registration invalidation, rejection/error
  fallback and shape/layout/mask/environment mismatches;
- same bytes but different shapes/dtypes, NaN/Inf, signed-zero exact checks,
  missing confidence fields, rigid transforms/reflections and degenerate arrays;
- fake workers: exit0 without output, out-of-order events, wrong run ID, invalid
  artifact path/hash, timeout, cancellation, child failure and monitor failure;
- CPU-only tests require no Torch/ROCm/model weights. GPU tests are explicit and
  include full outputs, route counts, precision and worker/source identities.

Existing runnable baseline: `PYTHONPATH=.:Protenix python -m pytest -q tests`.
Rust toolchain is available; after workspace creation use `cargo test --workspace`,
`cargo fmt --all --check`, and `cargo clippy --workspace --all-targets -- -D warnings`.
These Cargo commands are planned gates, not claims of a current Rust build.

## 9. Risk and Impact Analysis

The direct registry consumer `dispatch` and its triangle consumers require bridge
parity before replacing the Python policy. Text-confirmed CK/recycle registration
must also migrate. Dynamic imports/monkeypatches are not fully represented by the
graph. Key risks are GPU ordinal confusion, Python/C ABI lifetime, precision drift,
worker stdout pollution, process cleanup, unsafe result acceptance, duplicated
policy, changed cache/source identities and historical path breakage. Avoid broad
file moves before the runtime boundary is tested. Do not weaken old checks.

## 10. Files Expected to Change

| Paths | Work |
| --- | --- |
| New `Cargo.toml`, `Cargo.lock`, `crates/`, `configs/`, `python/`, `kernels/` | Rust framework, worker boundary, native sources |
| `fireprot/ops/registry.py` and AMD loaders | compatibility bridges after parity |
| model adapters and three benchmark entry points | extract model work; retire policy ownership |
| `scripts/infer.sh`, `scripts/sync_to_diamondhill.sh` | Rust entry point and checked deployment contents |
| `.gitignore`, README, architecture/context, migration docs, CI | build artifacts and truthful maintained status |
| tests and provenance | parity, failure cases, deployment/migration evidence |

## 11. Reusable Implementation Context

```json
{
  "implementation_context": {
    "task_summary": "Rebuild the maintained FireProt framework in Rust before resuming OpenFold3 optimization",
    "acceptance_criteria": [
      "Rust owns framework requests, dispatch policy, lifecycle, monitoring, provenance and validation",
      "Three existing model workflows pass parity",
      "Historical evidence and upstream checkouts preserved",
      "Document all remaining Python roles"
    ],
    "evidence_provenance": {
      "schema_version": 2,
      "head_commit": "c2dd7535aa652ff0e2e8d9a12d8e83709042edda",
      "generated_plan_path": "docs/plans/2026-10-04-gitnexus-plan-rust-framework-rebuild.md",
      "global_dirty_digest": {
        "algorithm": "sha256",
        "canonicalization": "gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records",
        "value": "0a9c85780067d9afcd0764f307b60891e3cee927ee11eaeb5ec7826d10fd82cd"
      },
      "cited_path_manifest": [
        {
          "path": ".gitignore",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:67c5acf74db5691225c0ce75a25665876ccd05cc2014dfc618947fe8146361ae",
          "index_digest": "sha256:67c5acf74db5691225c0ce75a25665876ccd05cc2014dfc618947fe8146361ae",
          "worktree_digest": "sha256:67c5acf74db5691225c0ce75a25665876ccd05cc2014dfc618947fe8146361ae",
          "untracked_digest": "absent"
        },
        {
          "path": "AGENTS.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:be8a35ef1baf2d3d3f21c5843daae1a1af54db7ba0eed71bdfd7311245c461ff",
          "index_digest": "sha256:be8a35ef1baf2d3d3f21c5843daae1a1af54db7ba0eed71bdfd7311245c461ff",
          "worktree_digest": "sha256:be8a35ef1baf2d3d3f21c5843daae1a1af54db7ba0eed71bdfd7311245c461ff",
          "untracked_digest": "absent"
        },
        {
          "path": "README.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:2978bef25ef88691e523b920ee57a5df87fc2f9017178b9bcd2fee85231c0236",
          "index_digest": "sha256:2978bef25ef88691e523b920ee57a5df87fc2f9017178b9bcd2fee85231c0236",
          "worktree_digest": "sha256:2978bef25ef88691e523b920ee57a5df87fc2f9017178b9bcd2fee85231c0236",
          "untracked_digest": "absent"
        },
        {
          "path": "benchmarks/atlasfold/profile_inference.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:90af3763bea56eb1fd45eb8cf1b8e252c2939aa467740a404e55a9d561e4fc52",
          "index_digest": "sha256:90af3763bea56eb1fd45eb8cf1b8e252c2939aa467740a404e55a9d561e4fc52",
          "worktree_digest": "sha256:90af3763bea56eb1fd45eb8cf1b8e252c2939aa467740a404e55a9d561e4fc52",
          "untracked_digest": "absent"
        },
        {
          "path": "benchmarks/openfold3/profile_inference.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:7b3c7b58a09001e421afc8f3a2309096edc6afbc3cb0716afc0f94fe34154bdb",
          "index_digest": "sha256:7b3c7b58a09001e421afc8f3a2309096edc6afbc3cb0716afc0f94fe34154bdb",
          "worktree_digest": "sha256:7b3c7b58a09001e421afc8f3a2309096edc6afbc3cb0716afc0f94fe34154bdb",
          "untracked_digest": "absent"
        },
        {
          "path": "benchmarks/protenix/profile_inference.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:4ee824921e41bfe3842bc9f7bb6fc35e67fbe43eced78dcba4d57f4ad26b5dee",
          "index_digest": "sha256:4ee824921e41bfe3842bc9f7bb6fc35e67fbe43eced78dcba4d57f4ad26b5dee",
          "worktree_digest": "sha256:4ee824921e41bfe3842bc9f7bb6fc35e67fbe43eced78dcba4d57f4ad26b5dee",
          "untracked_digest": "absent"
        },
        {
          "path": "docs/architecture.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:4d4cbbc9be16d11f554292e3a361b8fcabb6334e50dabcfb23a35735838d45ba",
          "index_digest": "sha256:4d4cbbc9be16d11f554292e3a361b8fcabb6334e50dabcfb23a35735838d45ba",
          "worktree_digest": "sha256:4d4cbbc9be16d11f554292e3a361b8fcabb6334e50dabcfb23a35735838d45ba",
          "untracked_digest": "absent"
        },
        {
          "path": "docs/rust-migration.md",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:9f82901a4fadafdb48561cfdbf74cebf0ea1f00b2145eee822efcd85457b38e2",
          "index_digest": "sha256:9f82901a4fadafdb48561cfdbf74cebf0ea1f00b2145eee822efcd85457b38e2",
          "worktree_digest": "sha256:9f82901a4fadafdb48561cfdbf74cebf0ea1f00b2145eee822efcd85457b38e2",
          "untracked_digest": "absent"
        },
        {
          "path": "fireprot/ops/registry.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:27a4b19d9dcbb6ebbb0e274790d3070dd79edb112c87345a5ccf8f7179637071",
          "index_digest": "sha256:27a4b19d9dcbb6ebbb0e274790d3070dd79edb112c87345a5ccf8f7179637071",
          "worktree_digest": "sha256:27a4b19d9dcbb6ebbb0e274790d3070dd79edb112c87345a5ccf8f7179637071",
          "untracked_digest": "absent"
        },
        {
          "path": "fireprot/validation/protenix.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:cdf3279f708b147ec014af3ec5970e3c7920d325533763fb7cbf01faa3f1d61b",
          "index_digest": "sha256:cdf3279f708b147ec014af3ec5970e3c7920d325533763fb7cbf01faa3f1d61b",
          "worktree_digest": "sha256:cdf3279f708b147ec014af3ec5970e3c7920d325533763fb7cbf01faa3f1d61b",
          "untracked_digest": "absent"
        },
        {
          "path": "scripts/infer.sh",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:90402e47cc4bcbfe99aab5a060ddcc0447b6023d5ffb0183a21800147870a590",
          "index_digest": "sha256:90402e47cc4bcbfe99aab5a060ddcc0447b6023d5ffb0183a21800147870a590",
          "worktree_digest": "sha256:90402e47cc4bcbfe99aab5a060ddcc0447b6023d5ffb0183a21800147870a590",
          "untracked_digest": "absent"
        },
        {
          "path": "scripts/sync_to_diamondhill.sh",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:b835cf43457486eb7570c0c493a34d47df61a79137b46fe74a12a64b759f0dde",
          "index_digest": "sha256:b835cf43457486eb7570c0c493a34d47df61a79137b46fe74a12a64b759f0dde",
          "worktree_digest": "sha256:b835cf43457486eb7570c0c493a34d47df61a79137b46fe74a12a64b759f0dde",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/benchmarks/test_protenix_profile.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:b8ba5131883b432ffac16bdaae578d29237930d19eb9610367106ab37d9ec59a",
          "index_digest": "sha256:b8ba5131883b432ffac16bdaae578d29237930d19eb9610367106ab37d9ec59a",
          "worktree_digest": "sha256:b8ba5131883b432ffac16bdaae578d29237930d19eb9610367106ab37d9ec59a",
          "untracked_digest": "absent"
        },
        {
          "path": "tests/operators/test_registry.py",
          "object_kind": {
            "head": "regular",
            "index": "regular",
            "worktree": "regular",
            "untracked": "absent"
          },
          "state": "clean",
          "rename_from": null,
          "rename_to": null,
          "head_digest": "sha256:8a7223f3efc8a24cb74fd93954451b6ba702c896e46e40a5e62cd66faaa908ea",
          "index_digest": "sha256:8a7223f3efc8a24cb74fd93954451b6ba702c896e46e40a5e62cd66faaa908ea",
          "worktree_digest": "sha256:8a7223f3efc8a24cb74fd93954451b6ba702c896e46e40a5e62cd66faaa908ea",
          "untracked_digest": "absent"
        }
      ]
    },
    "primary_symbols": [
      {
        "symbol": "OperationRegistry.resolve",
        "file": "fireprot/ops/registry.py",
        "lines": "130-160",
        "role": "dispatch policy"
      },
      {
        "symbol": "_GpuProcessMonitor",
        "file": "benchmarks/protenix/profile_inference.py",
        "lines": "199-262",
        "role": "isolation policy"
      },
      {
        "symbol": "compare_prediction",
        "file": "fireprot/validation/protenix.py",
        "lines": "129-184",
        "role": "numerical gate"
      }
    ],
    "related_symbols": [
      {
        "symbol": "OperationRegistry.dispatch",
        "relationship": "CALLS resolve",
        "relevance": "direct bridge consumer"
      }
    ],
    "execution_path": [
      "validate request",
      "check provenance/device",
      "run model worker",
      "verify outputs and monitor",
      "persist terminal manifest"
    ],
    "pdg_constraints": [
      {
        "description": "Monitor failure must veto isolation pass; source verified, bounded PDG corroboration",
        "affected_statements": [
          "benchmarks/protenix/profile_inference.py:223",
          "benchmarks/protenix/profile_inference.py:227"
        ],
        "implementation_consequence": "Keep unavailable/error states distinct from no other process"
      }
    ],
    "architectural_patterns": [
      {
        "pattern": "Native fallback when no implementation matches",
        "example_location": "fireprot/ops/registry.py:dispatch",
        "usage_guidance": "Preserve in Rust registry and bridge"
      }
    ],
    "files_to_modify": [
      {
        "file": "Cargo.toml",
        "symbols": [],
        "intended_change": "new workspace"
      },
      {
        "file": "crates/",
        "symbols": [],
        "intended_change": "new core/runtime/validation/CLI/bridge"
      },
      {
        "file": "python/",
        "symbols": [],
        "intended_change": "model-worker boundary"
      },
      {
        "file": "fireprot/ops/registry.py",
        "symbols": [
          "OperationRegistry"
        ],
        "intended_change": "bridge after parity"
      },
      {
        "file": "docs/architecture.md",
        "symbols": [],
        "intended_change": "actual Rust architecture"
      }
    ],
    "tests": [
      {
        "file": "crates/fireprot-core/tests/contracts.rs",
        "scenarios": [
          "Reject incompatible schema/model settings",
          "Round-trip valid workload without changing recycles or steps"
        ]
      },
      {
        "file": "crates/fireprot-runtime/tests/worker.rs",
        "scenarios": [
          "Worker exit0 with missing output fails",
          "Unexpected GPU PID fails",
          "Cancellation cleans up worker"
        ]
      },
      {
        "file": "crates/fireprot-validation/tests/parity.rs",
        "scenarios": [
          "dtype/shape/field mismatch fails",
          "Exact and established aligned comparison fixtures agree"
        ]
      }
    ],
    "verification_commands": [
      "PYTHONPATH=.:Protenix python -m pytest -q tests"
    ],
    "planned_verification_commands": [
      "cargo test --workspace",
      "cargo fmt --all --check",
      "cargo clippy --workspace --all-targets -- -D warnings"
    ],
    "risks": [
      "GPU mapping",
      "ABI lifetime",
      "precision drift",
      "duplicated Python policy",
      "artifact trust",
      "historical source hashes"
    ],
    "assumptions": [
      "Rust dependency and remote toolchain availability must be verified",
      "Policy bridge feasibility needs a prototype"
    ],
    "open_questions": [],
    "avoid": [
      "Do not restart completed baseline work",
      "Do not reduce model work or weaken numerical gates",
      "Do not alter independent upstream Git structure",
      "Do not resume OpenFold3 tuning before migration validation",
      "Do not claim Rust itself accelerates GPU inference"
    ]
  }
}
```

## 12. Assumptions and Open Questions

- [assumed] Existing CPU environments can build the selected Rust dependency set;
  verify with Cargo before pinning dependencies. Remote Rust build availability
  must be checked; do not modify shared model environments to install it.
- [assumed] A small C ABI policy bridge is sufficient without porting upstream
  Torch graphs; prototype and validate rather than treating this as proven.
- Remaining model-specific Python responsibilities must be inventoried and
  disclosed. Native GPU tensor execution and upstream model definitions are an
  explicit boundary, not an excuse to leave framework orchestration in Python.
- Deferred until after migration: new OpenFold3 operators, OpenFold2/ESMFold
  adapters, distributed tensor parallelism, and new approximation policies.
- No user input currently blocks implementation. Follow existing authorization
  for reversible edits, tests, remote validation, commits and pushes.

## 13. Definition of Done

The maintained Rust CLI runs all three currently supported model workflows with
explicit configurations, independently verified artifacts, device isolation,
backend identity and preserved scientific semantics. Rust owns framework policy;
Python is limited to listed upstream/tensor glue. Existing promoted HIP/CK paths
and native fallback pass operator and full-model parity. Tests/build/CI and
examples run, repository layout and docs reflect actual capabilities, historical
artifacts remain reproducible, and final commits are pushed. No pending required
model validation or hidden legacy framework dependency remains.
