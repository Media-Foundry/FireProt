# OpenFold3 qualified-source deployment audit

Status: source audit, 2026-10-10. The inventory below covers the frozen
isolated-data series. Unroll4 QKV fusion has since passed paired E2E; packed global
diffusion bias reuse passed whole-model gates but failed the all-pairs-faster
promotion criterion. HIP packing plus grouped global softmax has passed its model
gate and is undergoing paired E2E.
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
for a later direct comparison. It is currently **unqualified through GPU execution**; the gated follow-up
coordinator described below waits for the incremental E2E decision.
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


## Frozen direct-native comparison workflow

The experimental preparation/qualification/collection scripts now make the
router usable in a bounded follow-up study. They have **not yet produced a
new direct-native result**. The coordinator waits for the HIP packing study to
finish and requires all three incremental pairs to be faster before preparing
this candidate's direct comparison.

`prepare_native_comparison.py PAIRED_ROOT DESTINATION OUTPUT_ROOT REFERENCE_WORKER`
validates six exact outputs, the paired performance gate, source-file hashes
and the complete copied-file inventory. It rejects added/removed source files
and a modified native reference. The destination must be unused. Both original
worker bodies remain byte-identical, with a common routing entry and explicit
source manifest. Both profiles share this worker tree and environment inventory;
all nine optimization selection flags are disabled in the native profile. The
candidate retains its already-qualified absolute kernel locations, which remain
external pinned dependencies. This is a frozen experiment, not a portable
installation or a promotion into the maintained profile.

The preparation writes two full-model qualification specifications and three
AB/BA/AB pairs. It declares backend, profile, optimization flags and the actual
Rust bridge file identity as comparison differences. Scientific workload stays
strict FP32 N1536, R3/four trunk passes, D200/S1, seed101 on HIP7. The native
worker preserves upstream data-module defaults; the candidate retains its
qualified isolated single-item data path. The numerical gates must show that
this execution difference preserves all scientific outputs.

`qualify_native_comparison.py DESTINATION` runs the two model gates through Rust
`measure-inference`. It requires all 11 arrays and complete CIF bytes to equal
the qualified anchor, verifies the selected route and its sealed output hash,
and rejects optimization receipts on the native route. It verifies candidate
coverage and permits only the recorded `data_module_args.num_workers=0` runner
difference. A route file merely present on disk is insufficient: it must appear
in the accepted output archive and retain its recorded bytes.

Only a successful `route-qualification.json` permits Rust `benchmark-pairs`.
`collect_native_comparison.py DESTINATION` then rechecks all six current output
archives, selected routes, runner settings, dispatch counts, all 11 arrays and
complete CIF bytes. It derives model medians from raw measurements and checks
E2E medians against the Rust comparison. No run is discarded.

Twelve CPU tests cover routing, native source isolation, profile selection,
changed source inventories, a regressed pair, and missing/tampered sealed route
receipts. GPU correctness and performance still require the full workflow.

Current coordinator: `outputs/experiments/openfold3-direct-native-followup-20261010`.
Reserved study destination: `outputs/experiments/openfold3-direct-native-latest-20261010`.
At preparation time the coordinator is waiting on the existing incremental
paired process; it has not launched another GPU workload.
