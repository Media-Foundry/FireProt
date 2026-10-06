# Protenix Base integration

Base-default v0.5.0 uses the existing Rust request and supervised Protenix worker
boundary. Its checkpoint passes strict CPU loading: 368,088,191 parameters,
4,092 FP32 tensors, no missing or unexpected keys. See
[checkpoint evidence](../provenance/protenix_base_checkpoint.json).

## Initial workload

The qualified N600 request and prepared N1000/N1536 requests select FP32, C10/D200/S1, seed 101,
sequence-only monomers without MSA or templates, on HIP device 7. Base's upstream
default sample count is **five**; these single-sample experiments are not the full
upstream default workload. Keep all scientific settings identical within each
native/FireProt comparison. Mini C4/D1 speedups do not establish Base performance.

```bash
target/release/fireprot inspect configs/requests/protenix-base-n600-native.json
target/release/fireprot infer configs/requests/protenix-base-n600-native.json \
  configs/diamondhill/protenix-base-init-only.json
target/release/fireprot infer configs/requests/protenix-base-n600-fireprot.json \
  configs/diamondhill/protenix-base-init-only.json
```

Copy requests with fresh run IDs and output paths before repeating them. The
N1536 equivalents use `protenix-base-n1536-{native,fireprot}.json`.
The same profile can be used for both arms: the native backend does not install
FireProt hooks. Only HIP devices 6 and 7 are authorized for these experiments.

## Transplant boundaries

Existing HIP/CK triangle multiplication, triangle attention, LayerNorm, transition
and layout paths retain their exact shape/dtype/stride/device capability guards.
Unsupported signatures retain native execution. Their selection on Base must be
verified in `operator-runtime.json`, and model outputs must be evaluated under the explicitly recorded policy.
The later user-authorized Base coordinate tolerance is described below; historical
comparison receipts retain their original coordinate/confidence gates.

Fast model construction and skipped template preparation remain disabled in the
original broad-operator Base profile. The separate construction experiment qualifies
Base initialization; template omission still only qualifies Mini. Base's deeper stack
and full diffusion workload require independent output validation. The first GPU transplant failed the original coordinate gate. See the dated
results and the subsequent user-authorized coordinate assessment below.


## First N1000 transplant: rejected

The native and candidate runs both load and produce accepted finite output
bundles, but the paired correctness gate rejects the candidate. Full invocation
times are 419.053 s and 185.192 s; model times are 349.312 s and 115.087 s.
These are diagnostic timings, **not an accepted speedup**. Aligned coordinate
RMSD is 0.0467943 Angstrom, above the unchanged 0.001 gate. Summary confidence
maximum delta is 0.000518799. Atom metadata and CIF atom identities match.
CK attention and HIP packed triangle multiplication each execute 1,048 times.
[Evidence](../provenance/protenix_base_first_transplant.json).

The identical native repeat also failed: aligned RMSD 0.00828346 Angstrom.
The first measurements use upstream nondeterministic execution, so they cannot
isolate the cause of the candidate difference. The maintained Base worker now
requires strict deterministic algorithms and uninitialized-memory filling in
both backends and checks this again at the model boundary. Per the updated experimental scope, use approximately 600 residues
for focused diagnostics and N1000 for E2E qualification. N1536 examples are
prepared but are not required or being run in this round.


## Fast construction qualification

The CPU qualification script in `experiments/protenix_base/check_initialization.py`
compares full Base construction, strict checkpoint loading, all state-dict tensors,
all named buffers, and Python/NumPy/Torch CPU random states. It passes; construction
alone takes 50.222 s natively versus 3.635 s with deferred initializers. This is not
an E2E measurement. The existing complete-checkpoint check and native recovery
remain mandatory. [CPU evidence](../provenance/protenix_base_cpu_initialization.json).

The strict N600 experiment isolates this optimization, disables GPU operator
substitutions, and retains C10/D200/S1. All nine arrays are bitwise equal;
summary confidence and raw coordinate differences are zero. CIF atom identities
match. Complete invocation time is **179.350 -> 127.047 s (1.412x)**, from one
serial pair with fresh processes and unflushed filesystem caches. This is not a
three-pair timing trend or a steady-state model acceleration claim. The runtime
confirms fast construction selected once, no recovery, and all 1,048 triangle
attention/multiplication calls in each family using native implementations.
[Full result](../provenance/protenix_base_strict600_initialization.json).

Use the qualified initialization-only profile for this scope:

```bash
target/release/fireprot infer configs/requests/protenix-base-n600-native.json \
  configs/diamondhill/protenix-base-init-only.json
target/release/fireprot infer configs/requests/protenix-base-n600-fireprot.json \
  configs/diamondhill/protenix-base-init-only.json
```

The input is an explicitly documented 600-residue prefix of the N1000 development
sequence. It is a performance/repeatability diagnostic, not an experimental
structure-quality benchmark. N1000 strict operator qualification remains open;
N1536 is outside this round's requested scope.

## Strict N1000 attention isolation

A subsequent strict deterministic N1000 C10/D200/S1 comparison enables only CK
attention and qualified initialization, leaving triangle multiplication and the
other GPU substitutions native. CK attention executes 1,048 times, but the
candidate still fails: aligned RMSD 0.0749593 Angstrom and summary confidence
maximum delta 0.00131607. Diagnostic invocation times are 430.648 s and
216.682 s; **no performance verdict is accepted**.
[Evidence](../provenance/protenix_base_strict1000_attention.json).
The completed initialization-only control matches all nine native arrays exactly.
Its invocation time is 357.921 s and model time is 333.301 s.

An experimental HIP bias/softmax kernel retaining native QK/PV arithmetic passes
30 synthetic bitwise cases at N600/N1000. This establishes only an operator screen;
trained-attention replay and complete model qualification remain outstanding.
[Screen evidence](../provenance/protenix_base_bias_softmax_screen.json).


## Full CK/HIP cycle and revised coordinate assessment (2026-10-07)

The user authorized **Kabsch-aligned all-atom RMSD <=0.1 Angstrom** for Base
numerical regression, compensating global rotation and translation. This does
not mean every atom is within 0.1 Angstrom. The diagnostic Rust example
`crates/fireprot-validation/examples/base_drift.rs` also measures relative CA
pair distances. Its tests distinguish rigid motion from internal deformation.
It reports a coordinate verdict in JSON; process success alone is not acceptance.
The generic CLI's historical 0.001 coordinate/confidence gates are unchanged.

All rows below use the same N1000 input, Base FP32 C10/D200/S1, seed 101,
strict determinism and memory filling, no MSA/templates, HIP7, fresh processes
and unflushed filesystem/kernel caches. Complete invocation timing includes
startup; model timing is a separate boundary.

| Execution | Complete invocation (s) | Model forward (s) |
| --- | ---: | ---: |
| Native computation with fast initialization (incremental baseline) | 357.921 | 333.301 |
| CK attention only with fast initialization | 216.682 | 192.641 |
| Full CK/HIP with fast initialization | 146.035 | 121.084 |
| Full CK/HIP repeat | 146.199 | 121.541 |

The full route is **2.451x** against the 357.921 s baseline and **1.484x**
against CK attention alone (32.6% less time). These are observations from one
baseline and two candidate runs, not a three-pair trend. A separate raw-native
initialization run took 430.648 s; its 2.949x comparison includes initialization
and must not replace the incremental baseline without explicit labeling.
The 430.648-to-357.921 difference also includes run-to-run model timing variation;
it is not an isolated measurement of initialization savings.

Full CK/HIP selects 1,048 CK attention and 1,048 packed triangle multiplication
calls. The two full runs match all nine arrays bitwise. Both have aligned all-atom
RMSD **0.054688 Angstrom** relative to the native/init-control prediction, passing
the new coordinate criterion. CA aligned RMSD is 0.043968 Angstrom; CA pair-distance
RMS delta is 0.033756 Angstrom, p95 absolute delta 0.034152 and maximum 0.838485.
The maximum shows that small aggregate RMSD does not bound every local change.
Summary confidence maximum delta is **0.00192642**: the legacy confidence gate
still fails and is not claimed passed. Atom identities and metadata match.

These metrics assess numerical changes to a prediction, **not accuracy against
experimental structures**. The performance sequence has no experimental truth.
The CA-only lDDT against the native prediction is 0.999659; it must not be labeled
experimental lDDT. Experimental-structure validation remains outstanding.
See [cycle evidence](../provenance/protenix_base_full_user_coordinate_gate.json).
Frozen profiles and receipts are retained in the evidence root recorded there;
this experiment does not automatically enable fast initialization in the broad
checked-in Base profile. The alternative native-BMM/HIP-softmax adapter remains
an undeployed prototype; its synthetic screen is not model qualification.

## Module profile and first HIP Graph experiment

A frozen worker with temporary GPU event hooks reproduces all nine full-CK output
arrays exactly. Inclusive module intervals are: trunk Pairformer **64.642 s**
(10 calls), diffusion denoiser **25.846 s** (200 calls), MSA module **11.712 s**
(10 calls). Within the trunk, triangle attention accounts for 30.525 s,
triangle multiplication 21.879 s and pair transition 8.543 s. Parent/child
intervals overlap and include CPU launch gaps; these are diagnostic timings,
not an uninstrumented E2E result. [Profile](../provenance/protenix_base_stage_profile.json).

The maintained Base route did not use graph capture. The first isolated HIP
Graph experiment captures the complete 48-block trunk on the second actual
cycle, after warming on the first, and replays with refreshed inputs for all
remaining cycles. It preserves C10/D200/S1 and includes capture in E2E timing.

| Full CK/HIP treatment | E2E (s) | Model (s) |
| --- | ---: | ---: |
| Graph off | 147.098 | 122.691 |
| Pairformer graph on | 145.534 | 120.949 |

One candidate-then-control pair measures **1.0107x**, only 1.06% less E2E time.
All nine arrays match bitwise, including full confidence arrays. Rust paired
comparison passes the unchanged numerical gates for graph-on versus graph-off;
this does not change the separate native-versus-CK confidence result. Capture
alone costs 7.675 s, with one capture and nine replays. Peak allocated memory is
10,069,061,632 bytes. There is no repeated timing trend or production promotion.
[Receipts](../provenance/protenix_base_recycle_graph.json).

The subsequent denoiser graph experiment targets its 200 repetitions, retaining
random augmentation/noise/sampler updates outside capture. It completed all 200 calls (one capture, 199 replays), and all nine arrays
match its same-worker graph-off control and the earlier full-CK output bitwise.
Rust paired comparison passes the unchanged gates. E2E is **145.258 -> 148.687 s**
(off -> on), or **0.9769x**, 2.36% slower in this one candidate-then-control pair.
Model time is 120.846 -> 123.636 s; capture costs 0.0953 s. This candidate is
not promoted: it offers no observed E2E gain. This does not establish that all
HIP Graph partitions are ineffective. [Denoiser receipts](../provenance/protenix_base_diffusion_graph.json).
See
[experimental entry points and constraints](../experiments/protenix_base/README.md).

## CK tile rescreen (2026-10-07)

The vector-bias kernel was rescreened under strict FP32 at N1000. Alternating
variant order, four warmup rounds and 16 measured rounds per variant retained
output allocation and deterministic memory filling. Bias was already contiguous.
Random and large-bias cases pass 2e-5 absolute/relative agreement with maintained
CK; this is a synthetic operator screen, not trained-model accuracy evidence.

| M / N / waves | Median (ms) |
| --- | ---: |
| 128 / 32 / 4 (current) | 20.008 / 20.030 across the two rounds |
| 64 / 32 / 4 | 26.616 |
| 64 / 64 / 4 | 22.136 |
| 128 / 64 / 4 | 20.925 |
| 256 / 32 / 8 | 21.852 |
| 128 / 128 / 4 | 23.505 |
| 256 / 64 / 8 | 32.648 |

Retain the current tile; no alternative is promoted. See
[first round](../provenance/protenix_base_ck_tiles.json) and
[larger tiles](../provenance/protenix_base_ck_tiles_wide.json).

## Published artifact loading

`configs/diamondhill/protenix-base-published.json` enables qualified fast
initialization and pins the existing AMD operators and CK attention as published
Rust-verified bundles. It retains template preparation and the original model
workload. The libraries are byte-identical to those used by the full-CK baseline;
publishing moves compiler-input inspection out of inference, while per-load
manifest/library/receipt hashes and runtime compatibility are still checked.

Copy `configs/requests/protenix-base-n1000-fireprot.json` with a fresh run ID and
output directory, then use the published profile with Rust `infer`. The bundles
are DiamondHill deployment assets under `runtime/kernels/`, not files shipped in
Git. Rebuild/publish and update pins explicitly on another host; see
[kernel publication](kernel-artifacts.md).

The first candidate-then-control pair measures **146.264571 -> 137.239499 s**
complete invocation time (**1.06576x**, 6.17% less time). Model-call timing is
121.883916 -> 112.447313 s; that boundary includes lazy operator loading, so this
is not a steady-state GPU compute improvement. Both published loaders are
confirmed selected, with 1,048 CK attention and 1,048 packed multiplication calls.
All nine arrays are bitwise equal and Rust paired comparison passes unchanged
gates between the two loading routes. The earlier native-versus-CK confidence
limitation still applies. [First-pair evidence](../provenance/protenix_base_published_first_pair.json).

The initial pilot used a 7200-second invocation deadline, while the repeated
jobs use 1800 seconds. Rust rejected combining those timing conditions. That
pilot is retained separately; an additional matching pair completed the final
**three AB/BA/AB pairs**, with all invocation deadlines set to 1800 seconds.
[Aggregation rejection](../provenance/protenix_base_published_aggregation_rejection.json).

| Three-pair median | Build-cache loading | Published loading | Speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 146.146473 s | 133.436657 s | 1.09525x |
| Model-call boundary (includes lazy loading) | 121.802534 s | 109.062242 s | 1.11682x |

Complete invocation time falls **8.70%**. Every individual pair is faster
(1.08579x to 1.09998x); Rust reports faster median and p95. All six archives match
all nine arrays bitwise, with unchanged published-route selection counts and
passing Rust reference/pair/aggregate gates. These descriptive repetitions do not
establish a cross-host or cold-cache guarantee. All trials retain strict FP32
N1000 C10/D200/S1 seed101, HIP7, fresh processes and unflushed filesystem/compiled
kernel caches. [Complete result](../provenance/protenix_base_published_three_pairs.json).

## Padded N1000 hipBLASLt candidate

A separate C ABI experiment preserves the current triangle input strides
`[1000128,1000,1]` and uses a contiguous output. Sixteen hipBLASLt heuristic
algorithms in each direction passed synthetic FP32 comparison to native hipBLAS.
Heuristic index 0 was fastest in both directions: incoming **9.248 -> 7.857 ms**,
outgoing **10.558 -> 7.905 ms**. Incoming matches bitwise; outgoing maximum
absolute delta is 3.28e-7 on this input. Output allocation and strict memory
filling are included; a reusable 32 MiB workspace is outside timing.

This is an operator screen, not model/E2E evidence. The full Base experiment is
running with the same C10/D200/S1 workload and a pinned published Lt bundle in
an isolated worker snapshot. It must pass aligned-coordinate/relative-geometry
checks and complete E2E comparison before promotion. The maintained N1536-only
Lt adapter and the published Base profile remain unchanged.
[Screen evidence](../provenance/protenix_base_lt_padded_screen.json).
