# Protenix Base optimization qualification audit

Status: final direct-native timing of the template-enabled maintained entry is
running. This audit does not yet declare the optimization cycle complete.

## Scope and acceptance

The qualified target is Base-default v0.5.0, FP32, 1,000 tokens, ten cycles,
200 diffusion steps, one sample and seed 101 on gfx90a. The performance input
has 7,979 atoms; a second 1,000-token input, 5ZMW, has 7,711 atoms. Measurements
use HIP device 7, serial fresh processes, strict determinism with memory filling,
published kernel artifacts and unflushed filesystem caches. These results do not
qualify arbitrary lengths, precision modes, sample counts or multi-GPU execution.

The native baseline is the native Torch route in the pinned Protenix source
manifest `bb3be993`, with the same qualified fast initialization. It is not a
separately measured pristine upstream installation. The historical 357.921 s
control used FireProt dispatch with native triangle operators; see the
[baseline route audit](../provenance/protenix_base_native_baseline_route_audit.json).

Correctness means prediction regression under the user-authorized Kabsch-aligned
all-atom RMSD limit of 0.1 Angstrom, unchanged summary-confidence tolerance of
0.001, and exact atom identities/metadata. Rigid displacement is removed before
the coordinate criterion. This is not a maximum per-atom error guarantee or a
claim of experimental structure accuracy. The generic Rust 0.001 Angstrom gate
remains unchanged; its failure is preserved alongside the separate user policy.

## Requirements and evidence

| Requirement | Inspected evidence | Current result |
| --- | --- | --- |
| Connect Base weights through the maintained Rust entry | [Request](../configs/requests/protenix-base-n1000-fireprot.json), [profile](../configs/diamondhill/protenix-base-n1000-optimized.json), [entry guide](protenix-base.md#maintained-n1000-optimization-entry) | Connected; model/weight/source/input pins explicit |
| Preserve cycles, steps, samples and precision | Maintained `base_n1000.validate_contract`, request and invocation records | Exact FP32 N1000 C10/D200/S1 contract; unsupported workloads rejected when enabled |
| Use CK/HIP and optimize the measured large stages | [Triangle profile](../provenance/protenix_base_triangle_profile.json), [promotion map](../provenance/protenix_base_promotion_map.json) | CK attention, scoped Lt, dual storage, diffusion bias caches and HIP transition products composed |
| Verify every promoted change | Versioned per-trial evidence linked in the [Base guide](protenix-base.md) | Rejected numerical/performance trials retained; no generic gate relaxed |
| Verify the actual maintained entry on two inputs | [Template-enabled entry qualification](../provenance/protenix_base_maintained_templates_qualification.json) | All nine output arrays bitwise equal to each input's prior optimized anchor; actual skip counters/source guard verified |
| Guard activation and restore hooks on failure | `tests/adapters/test_protenix_base_n1000.py`, `test_protenix_features.py`, `test_protenix_base_template_experiment.py`, `test_protenix_trunk_graph_contract.py`, `tests/workers/test_protenix.py` | 70 targeted CPU checks pass; GPU entry qualification supplies the numerical evidence |
| Report direct complete-invocation E2E | [Earlier three native pairs](../provenance/protenix_base_maintained_native_three_pairs.json) | 3.42210x before template omission; updated direct series still running |
| Document entry and measured limitations | [README](../README.md), [Base guide](protenix-base.md) | Entry, workload, timing boundary and separate accuracy criteria documented |
| Work on master and publish changes | Commit `1b1825a`, pushed to `origin/master` | Template promotion published; final report/audit still pending |

The entry qualification is a migration/regression check, not a second-input
speedup measurement. Bitwise agreement with an optimized anchor does not imply
bitwise agreement with native predictions. The earlier native comparison has
aligned RMSD 0.006099824 Angstrom and summary-confidence maximum delta 0.0000495911;
full confidence arrays retain nonzero diagnostic differences.

## Evidence for diminishing returns

The following results use different baselines and must not be multiplied into
a direct native speedup. Synthetic screens are explicitly separate from E2E.

| Candidate | Observed result | Decision |
| --- | --- | --- |
| CK block scheduling | Best synthetic difference about 0.27%; all eight numerical checks exact | Keep current scheduling; [evidence](../provenance/protenix_base_ck_swizzle.json) |
| Dual-projection tile changes | Alternatives 3.020–4.004 ms versus 2.888 ms maintained; 14 exact checks | No replacement; [evidence](../provenance/protenix_base_dual_tiles.json) |
| Fused A/B projection | Best synthetic 5.773 versus 5.787 ms; ten exact checks | No promotion; [evidence](../provenance/protenix_base_quad_projection.json) |
| HIP diffusion transition fusion | Three-pair E2E median 112.610 to 112.041 s, 1.00508x; final pair only 0.086% faster | Retain qualified small gain; [evidence](../provenance/protenix_base_diffusion_transition_three_pairs.json) |
| Attention gate/layout fusion | Isolated operator improves, complete invocation 112.049 to 112.478 s | Reject E2E regression; [evidence](../provenance/protenix_base_diffusion_gate_first_pair.json) |
| Composed Pairformer HIP Graph | Three-pair median 112.761 to 113.375 s, 0.99459x; all nine arrays exact | Keep eager execution; [evidence](../provenance/protenix_base_maintained_vs_graph_three_pairs.json) |
| Unused-template preprocessing | Three-pair median 111.936 to 110.248 s, 1.01531x; all three pairs faster and all outputs exact | Promoted after second-input and maintained-entry qualification; [evidence](../provenance/protenix_base_unused_templates_three_pairs.json) |

These results establish diminishing returns for the tested implementation choices,
not a proof that all possible kernels are optimal. Persistent multi-request Graph
reuse, other sequence buckets and approximate recycle/step reduction remain
unqualified. No cycle or diffusion step was removed to obtain these results.

## Remaining completion gate

The final study is
`outputs/experiments/protenix-base-final-native-pairs-20261007` on DiamondHill.
It alternates native/optimized, optimized/native, native/optimized. Every run
must reproduce its own arm's prediction anchor, actually select fast initialization,
and, for the optimized arm, report one verified unused-template omission. Each
completed pair also checks rigid-aligned coordinates.

Before completion, require the terminal successful driver receipt and all twelve
journal events, authenticate all six measured invocations through Rust, retain
the legacy numerical report, evaluate the separate user policy, publish the new
direct median ratio, and update this audit from those completed records. A live
process or a passing subset of the six runs cannot satisfy that gate.
