# OpenFold3 qualified-source deployment audit

Status: source audit, 2026-10-10. No maintained-profile promotion or new GPU
qualification. The current stable experimental composition is the independently
qualified isolated-data series; QKV fusion is still undergoing paired timing.

## Evidence boundary

[Source mapping](../provenance/openfold3_qualified_source_map_20261010.json)
compares the frozen isolated-data source inventory with repository files and
records both hashes. It is an inventory comparison, not a trace of imported
modules. Frozen source root:
`/media/IntelSSD/fireprot/outputs/experiments/openfold3-isolated-data-v2-20261009`.
The model and three-pair evidence is
[recorded separately](../provenance/openfold3_isolated_data_e2e_n1536.json).

Most cache, workspace, normalization, gate, data-preparation and template helpers
already have byte-identical repository sources. Path differences remain explicit
because package-relative imports and runtime root discovery affect execution.
Frozen Protenix files and unrelated framework drift must not be copied as part
of this OpenFold3 integration.

## Remaining integration boundaries

| Boundary | Current evidence | Requirement before promotion |
| --- | --- | --- |
| Attention coverage | Frozen dispatcher explicitly enables D16/D32; maintained dispatcher accepts D32 | Explicit selection and truthful launch counters; strict unsupported-input fallback |
| Composition | Frozen worker composes owned scores, workspace, confidence chunk cap, pair/atom caches, HIP copies/gates, template reuse, normalization and isolated data | Preserve scope order, restoration on error, source pins and per-workload eligibility |
| Native control | Frozen outer caches/scopes can remain enabled even with a native attention backend | Disable all optimization scopes for a direct original-Torch comparison; the isolated-data model worker is unconditional, so its data path cannot be disabled by an environment flag alone |
| Tensor layout | Qualified normalization uses canonical output strides | Preserve the existing discriminating layout regression and full-model exactness |
| Artifact loading | Some helpers load pinned libraries from frozen roots | Publish/verify explicit Rust artifact identities; remove reliance on incidental experiment-directory layout |
| Workspace | Stream/thread ownership and budget are enforced by existing Rust leases | Retain lifecycle/ownership checks and independent outputs; identify any fused entry override separately |
| Output evidence | Frozen transport includes normalization/template/cache files, while collectors separately check newer receipts | Hash gate, key-packing, isolated-data and selected-fusion receipts in completed output bundles |
| Acceptance | N1536 strict FP32 R3/D200/S1 seed101, all 11 arrays and full CIF bytes | Re-run that gate through the exact promoted entry, then matched E2E; no claim for untested workloads |

The source audit does not select an unfinished kernel trial or authorize a lower
numerical standard. Final runtime defaults should follow completed paired E2E
results. Earlier original-Torch timings cannot be combined with newer candidate
observations to manufacture a direct speedup ratio.

The isolated-data **paired** worker has `FIREPROT_ISOLATED_DATA` selection; the
standalone model worker audited here always sets `num_workers=0` and enters the
isolated RNG scope. A future original-Torch control must use an explicitly
conditional/verified worker. Setting a flag that its source never reads is not
a native baseline.
