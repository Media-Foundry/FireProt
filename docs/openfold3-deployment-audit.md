# OpenFold3 qualified-source deployment audit

Status: source audit, 2026-10-10. The inventory below covers the frozen
isolated-data series. Unroll4 QKV fusion has since passed paired E2E; packed global
diffusion bias reuse has passed whole-model gates and is undergoing paired E2E.
Those later files are not implicitly covered by the older inventory. No
maintained-profile promotion has occurred.

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


## Direct-native entry preparation

The maintained `python/fireprot_workers/openfold3.py` has SHA-256
`c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e`.
Its native branch directly calls the captured upstream forward. It neither sets
`data_module_args.num_workers=0` nor installs pair/atom/template/bias caches,
normalization, gates, workspace or QKV scopes around native prediction. The
ordinary CLI configuration and output serialization are retained. This is a
source audit, not a new native performance measurement.

`experiments/openfold3_diffusion/native_router.py` prepares a common worker entry
for a later direct comparison. It is currently **unconnected to GPU execution**.
A frozen tree must contain:

| Worker-package file | Required contents |
| --- | --- |
| `openfold3.py` | The comparison router |
| `openfold3_reference.py` | Exact bytes of the audited maintained worker above |
| `openfold3_optimized.py` | One separately qualified optimized worker |
| `openfold3_routes.json` | Schema1; `workers` entries for `native` and `hip_fused`, each with fixed module name and SHA-256 |

Both arms share that tree. The router uses the request backend, verifies the
selected file digest and import origin, and lazily imports only that worker.
The native digest is also fixed in code, so a manifest cannot silently relabel
an optimized worker as the reference. The original request and channel are
forwarded without rewriting. A successful call writes `openfold3-route.json`
identifying the actual selected source. Five CPU tests exercise real imports,
unselected-module exclusion, source corruption and the fixed reference pin.
They do not establish model correctness or performance.

Before a direct-original claim, preparation must pin the complete shared tree,
include the route receipt and newer optimization sidecars in completed artifact
hashes, disable baseline optimization flags, and declare all actual backend/
profile/environment/bridge differences to Rust paired validation. Native data
preparation must retain its upstream defaults; an optimized worker with
unconditional data isolation is still unsuitable for that branch. Verify both
routes on the full workload, including all11 arrays and complete CIF bytes,
then run fresh matched E2E. The result will compare the pinned upstream native
Torch route under the common FireProt launcher with the selected composition;
it will not measure an unwrapped standalone upstream CLI.
