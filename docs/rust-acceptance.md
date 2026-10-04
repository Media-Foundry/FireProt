# Rust framework acceptance

Date: 2026-10-05. This audit evaluates the original
[Rust rebuild plan](plans/2026-10-04-gitnexus-plan-rust-framework-rebuild.md),
including its ownership requirements and Definition of Done. It covers the
maintained framework and documented model workflows, with the limits below.

## Requirement-to-evidence review

| Requirement | Implemented boundary and evidence | Finding |
| --- | --- | --- |
| Typed, model-specific workloads without reducing inference work | `fireprot-core` request/protocol modules; request contract tests; retained installed-run requests | Pass; Protenix cycles and other models' recycles remain distinct |
| Rust registry policy with native fallback | `fireprot-core::registry`, CPython bridge, registry/bridge tests; `docs/rust-operators.md` | Pass; Python supplies tensor metadata and capability callbacks |
| Rust process lifecycle, protocol, deadlines and physical GPU isolation | Runtime supervisor, device monitor, process cleanup and failure tests; independently verified devices in all installed reports | Pass; workers cannot authorize their own isolation |
| Source, weight, input and environment provenance | Rust preflight, source export, execution records and expanded inventories; `rust_system_environment_integration.json` | Pass for declared deployment inputs; no hermetic-environment claim |
| Numerical and structure acceptance | Rust NPY/NPZ, metrics, CIF and bundle validators; AtlasFold schema 3 input/atom14 binding and relocation/tamper tests | Pass under model-specific gates |
| Three existing model workflows | Seven installed requests with reference comparisons and archive replay in `rust_installation_v2_integration.json` | Pass for the README matrix |
| Preserve promoted operators and fallback | Six Rust-controlled native loaders, published artifacts; layout GPU checks and native/accelerated model records | Pass; no kernel arithmetic changed during framework rebuild |
| Rust native build/cache ownership | Rust build identities, dependency hashes, locks, compiler guard, cancellation and atomic publication; build/process tests; six installed cold-loader checks in `rust_installation_integration.json` | Pass; Torch-owned descriptors and compilation calls remain explicit Python boundaries |
| Maintainable layout and preserved history | Five Cargo crates, `python/`, `kernels/amd`, root compatibility shim; `layout_path_map.json` and `entrypoint_path_map.json` | Pass; independent upstream checkout and ignored runtime/output assets preserved |
| Maintained command interfaces | Rust infer, batch, measurement, collection and paired commands; shell wrappers delegate with `exec`; batch and entrypoint tests | Pass; historical benchmark scripts are reproduction tools |
| Timing and correctness-gated benchmark workflow | `rust_invocation_timing_integration.json`, `rust_paired_comparison_integration.json`, `rust_paired_execution_integration.json` | Pass; full invocation and model intervals stay distinct |
| Independent installation | 58-file v2 prefix, seven runs from `/tmp`, batch execution, post-run manifest verification | Pass; external model assets remain pinned inputs |
| CPU tests and CI | 172 Rust tests on local default and Rust 1.85; 273 Python tests passed, 110 environment-dependent skips; remote stable/MSRV CI on `514e42b` and `66e064e` | Pass; GPU evidence is separate from skipped CPU tests |
| Documentation and remaining Python roles | README model matrix, architecture, `python/README.md`, installation and this audit | Reviewed; dated progress entries are historical |

Evidence filenames in the table are under `provenance/`. The installation evidence
retains exact requests, acceptance receipt hashes, supervisor/device reports,
comparisons, replay hashes, build inputs and the installation manifest. Earlier
operator/layout evidence remains attributable to its original source identities;
it is not relabeled as a new kernel measurement.

## Installed model acceptance

- AtlasFold plus AtlasLM-3B, fused HIP: N1000 and N1536; all public arrays exact.
  N1536 retains 100 diffusion steps and N1000 retains 30.
- Protenix Mini-default, FireProt: N1000 and N1536; unchanged coordinate and
  summary-confidence thresholds of 1e-3.
- Protenix Mini-ESM, FireProt: N1000; independent upstream CIF/summary gates and
  byte-identical FP32 `[1000, 2560]` ESM input features.
- Protenix Tiny-default, native: L124; independent upstream CIF/summary gates.
- OpenFold3/OpenBind-0, strict deterministic native: N1000; eleven arrays exact.

Each run also passes archive replay and monitored physical-device acceptance.
The batch covers Tiny, Mini-default N1000 and AtlasFold N1000 sequentially.
Other previously verified native/chunked/HIP routes retain their separately linked
migration records. This final installed matrix does not claim every possible
model, precision, length or sample-count combination.

## Environment coverage and cost

The expanded inventories include declared Python/base-interpreter roots, ROCm,
system libraries/locales, loader configuration/cache, OS/kernel/AMDGPU identity
fields and OpenFold3's static cache configuration. Optional loader entries have
explicit absence checks. Published kernels and the installed bridge have separate
artifact verification. Observed regular mappings from the completed model workers
have no unaccounted paths; their PIDs match supervisor records.

This is declared-content verification, not a filesystem snapshot or interception
of all future/transient imports, GPU firmware or runtime plugins. Point-in-time
maps have that explicit limit. Inventories remain opt-in and rehash at checkpoints.
The expanded standalone captures took about 12.7–15.6 seconds under the recorded
host conditions. Scan reuse reduced an earlier AtlasFold capture median from
15.855 to 12.202 seconds; neither observation is inference E2E speedup.

## Compatibility limits and follow-up work

OpenFold3 legacy default-mode exact parity remains unpassed because unchanged
native repeats vary; only the explicit strict deterministic configuration is
accepted. No accelerated OpenFold3 route is promoted. Protenix full confidence
arrays remain shape/dtype/finite checked without a bitwise accelerated/native
claim. Tiny has no long-input performance claim. Long synthetic cases establish
native agreement, not independent experimental structural accuracy.

Python remains at upstream model execution, tensor predicates/calls, Torch build
descriptors and serialization. The small compiler-discovery calls in that boundary
provide descriptors to Rust; Rust owns hashing, lock/process/deadline policy and
publication. Historical Python validation helpers are not used for maintained
acceptance. OpenFold2, ESMFold, distributed execution and new approximation policies
are explicitly deferred in the original plan. New OpenFold3 tuning follows the
validated framework and must preserve correctness after every trial.

Final publication consists of committing this audit and installation evidence,
pushing the branch, and checking the resulting repository and CI state. Technical
acceptance does not substitute for that final publication check.
