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

The first full Base pair retains C10/D200/S1 and a pinned Lt bundle in an
isolated worker. E2E is **132.947 s off -> 134.390 s on**, about **1.09% slower**;
model-call time is 108.921 -> 109.776 s. All 524 incoming and 524 outgoing calls
select Lt with no fallback. This candidate is not promoted. A trained-call
profile is investigating why the operator-screen gain did not carry into E2E.

Aligned all-atom RMSD is 0.006100 Å against the native/init-control prediction
and 0.050504 Å against the previous CK/HIP output, both within the user 0.1 Å
criterion. CA pair-distance RMS deltas are 0.004659 and 0.029868 Å respectively.
Summary confidence maximum delta against the previous CK route is 0.00187683;
the old 0.001 coordinate/confidence comparison fails and its performance verdict
remains withheld. The timings above are diagnostic observations under the separate
user coordinate criterion, not a claim that legacy gates or experimental accuracy
passed. [Full-model evidence](../provenance/protenix_base_lt_first_pair.json). The maintained N1536-only
Lt adapter and the published Base profile remain unchanged.
[Screen evidence](../provenance/protenix_base_lt_padded_screen.json).

## Trained Lt cost diagnosis and Torch BLAS reuse

The instrumented contraction probe preserves all nine arrays for each route.
Summed inclusive contraction events fall from 10.385 s (native hipBLAS) to
8.900 s (direct Lt), but direct adapter setup costs 2.112 s and the first Lt calls
cost roughly 0.2–0.3 s each. Setup offsets the contraction improvement. These
inclusive event intervals include launch gaps and are not complete E2E timings.
[Profile evidence](../provenance/protenix_base_lt_profile.json).

Reusing Torch's `hipblaslt` preference around each matching BMM avoids loading
the additional wrapper library. The preference is restored in `finally` before
returning, preserving other operators' selection. A synthetic screen, including
preference-switch cost, measures incoming 9.256 -> 7.515 ms and outgoing
10.537 -> 7.567 ms. [Screen](../provenance/protenix_base_torch_blas_screen.json).

One matched full-model pair measures **132.950508 -> 130.926517 s E2E**
(**1.01546x**, 1.52% less time); model-call time is 108.905801 -> 106.646227 s.
All nine candidate arrays are bitwise equal to the direct-Lt candidate, hence
the same 0.006100 Å native-relative / 0.050504 Å previous-CK-relative aligned
RMSD assessment applies. Both directions select 524 calls without fallback.
The old coordinate/confidence comparison still fails; its withheld performance
verdict is preserved alongside these separate user-policy observations. This is
not a repeated timing trend or a production promotion.
[First pair](../provenance/protenix_base_torch_lt_first_pair.json).

Three compatible pairs (BA/AB/BA, all invocation timeouts 1800 s) now measure
complete invocation medians **133.855739 -> 131.122685 s**, **1.020843x** and
**2.04% less time**. Every pair is faster (1.01546x–1.02084x). Model-call medians
are 109.598212 -> 106.988373 s. All six runs match all nine arrays bitwise to
their corresponding first-run route, and all candidates select 524 incoming and
524 outgoing Lt contractions without fallback. These repeated timing observations
use the separate user 0.1 Å coordinate criterion; the unchanged legacy comparison
still rejects coordinate/confidence differences and withholds its performance
verdict. No default-route promotion follows from this report.
[Three-pair evidence](../provenance/protenix_base_torch_lt_three_pairs.json).


A separate full-forward Torch BLAS-preference pair measures 133.003585 ->
132.596816 s E2E (1.00307x, only 0.31% less time). Native-relative aligned RMSD
is 0.047952 Å, and previous-CK-relative RMSD is 0.009690 Å. Both meet the user
coordinate criterion. This small single-pair gain is not promoted; requesting a
preference does not prove every internal call uses that library.
[Global preference evidence](../provenance/protenix_base_global_blas_first_pair.json).

## Experimental structure qualification: 5ZMW chain A

A sequence-only 1000-residue development case is stored in
`benchmarks/protenix/experimental/5zmw_A`. Its metadata pins the source mmCIF,
canonical sequence, alternate-location selection, and per-residue mapping.
`gt_mapping.npz` contains **CA-only experimental `coordinates[mask]`**, with
1000 valid positions, rather than input reference conformations. Predictions
omit experimental ligands and state context; this case is not an independent
or training-excluded test set.

The Rust `experimental_ca` example checks artifact hashes, residue identities,
and NPZ/CIF coordinate binding, then reports Kabsch-aligned CA RMSD and CA-only
lDDT against experimental coordinates. It does not change acceptance policies.
One matched pair on this input uses FP32 C10/D200/S1, seed101, HIP7, no MSA
or templates, fresh processes and unflushed filesystem caches. Both arms retain
qualified fast initialization; the candidate combines published CK/HIP and scoped
Torch Lt (524 incoming and 524 outgoing contractions, no fallback).

| Metric | Native GPU computation | Optimized |
| --- | ---: | ---: |
| Complete invocation E2E | 357.848379 s | 130.295645 s |
| Model call | 333.304562 s | 106.450242 s |
| Experimental CA aligned RMSD | 30.349352 Å | 30.350175 Å |
| Experimental CA-lDDT | 0.31538224 | 0.31534813 |

This is **2.74643x E2E** in one pair, not a repeated trend. Numerical regression
between predictions gives all-atom aligned RMSD **0.063961 Å** (7711 atoms),
within the user 0.1 Å criterion; CA aligned RMSD is 0.041719 Å. Summary confidence
maximum delta is 0.000179291, below 0.001. The legacy 0.001 Å coordinate gate
still fails and its performance verdict remains withheld; that record is retained.
Experimental metrics show very little change, but both absolute predictions have
low structure agreement under this sequence-only setup. No claim of a generally
accurate or independent test result follows from this single case.
[Paired timing, numerical and experimental evidence](../provenance/protenix_base_5zmw_first_pair.json).

## Triangle multiplication substage profile

A scoped-Lt full-model run with `triangle_probe.py` preserves all nine arrays
bitwise against the uninstrumented candidate. Inclusive GPU-event intervals are:

| Substage | Calls | Total |
| --- | ---: | ---: |
| Packed BMM | 1048 | 8.355 s |
| Dual projection and packing | 2096 | 6.105 s |
| Triangle residual/gate fusion | 1048 | 1.780 s |
| Packed output normalization | 1048 | 1.679 s |
| Gate linear | 1048 | 1.358 s |
| Output linear | 1048 | 1.341 s |
| Input normalization | 1048 | 1.266 s |

These include launch gaps, hooks and allocation effects; they are diagnostic
intervals rather than attainable E2E savings. The next kernel target is the dual
projection, which remains the largest measured non-BMM triangle substage.
[Profile evidence](../provenance/protenix_base_triangle_profile.json).

## Dual-projection tile and storage experiments

The seven-tile HIP screen compares exact logical padded outputs against the
maintained projection at ordinary and enlarged gate scales. All 14 checks are
bitwise equal. Alternative row/channel/K/wave tiles take 3.020–4.004 ms versus
2.888 ms maintained. Recompiling the identical tile through a C ABI takes
2.866 ms; that small difference does not establish a new tile improvement.
[Tile screen](../provenance/protenix_base_dual_tiles.json).

A second screen fuses both A/B projections into one four-product kernel. All
10 numerical checks are bitwise equal. Two maintained calls take 5.787 ms;
the best fused variant takes 5.773 ms, while other variants take 7.540–10.443 ms.
Neither tile replacement nor A/B fusion is promoted.
[Fusion screen](../provenance/protenix_base_quad_projection.json).

Retaining initialized output storage reduces the synthetic steady-state call
from 2.885 to 2.482 ms (about 14% less time), with exact outputs for three
successively changed inputs. Deterministic initialization stays enabled;
initial storage allocation is excluded from this microbenchmark and must be
included in model E2E. [Reuse screen](../provenance/protenix_base_dual_reuse_screen.json).

`model_dual_reuse.py` experiments with two retained projection slots. It checks
FP32 N1000 shape, layout, device, stream, A/B input identity, and consumption by
the matching packed BMM before allowing reuse. It rejects capture and unexpected
call sequences. This remains an isolated frozen-worker trial; no maintained
operator or default profile has changed.

The first matched full-model pair rejects this reuse adapter: complete invocation
**131.688016 -> 133.046890 s** (about **1.03% slower**), with model-call time
107.224377 -> 108.595448 s. All nine arrays are bitwise equal, and unchanged
Rust comparison gates pass between these two routes. The candidate records one
retained allocation, 2096 projections and 1048 consumed pairs without error.
[First pair](../provenance/protenix_base_dual_reuse_first_pair.json).
A separate instrumented trial is separating common AMD loading, additional
adapter setup, and trained substage intervals; no default promotion is made.

### Receipt parsing diagnosis

The paired trained profile preserves all nine arrays. Retained dual-projection
intervals total 5.280 s versus 6.008 s with allocation, but extra adapter setup
costs 2.169 s. Common AMD extension preload alone costs about 6.75 s.
[Setup and trained profile](../provenance/protenix_base_dual_reuse_profile.json).

A local CPU probe identifies an unbuffered Rust JSON read in published-kernel
acceptance. Parsing the actual 2.4 MB build receipt makes 2,436,810 `File::read`
calls and takes about 1.65 s. `BufReader` reduces this to 299 reads and about
6 ms; all decoded receipt values match exactly across three alternating pairs.
[CPU evidence](../provenance/kernel_receipt_read_probe.json). These are local
parsing timings, not DiamondHill E2E measurements.

The runtime fix only wraps this file read in `BufReader`. Size limits, hashes,
schema and compatibility checks remain in place; all three kernel-bundle tests
pass. A DiamondHill E2E trial uses paired Rust bridges built from one source
snapshot differing only in this change. The buffered candidate completes in
122.355360 s E2E (98.130984 s model call), with all nine arrays bitwise equal
to the earlier scoped-Lt output. The matched unbuffered control takes
131.150439 s E2E (106.907067 s model call), giving **1.071881x** and **6.71% less
time** in the first pair. All nine arrays match bitwise and unchanged Rust
comparison gates pass.
[First paired evidence](../provenance/protenix_base_buffered_receipts_first_pair.json).


Three compatible BA/AB/BA pairs now confirm complete-invocation medians
**131.744855 -> 122.458591 s**, **1.075832x**, or **7.05% less time**. Baseline
range is 131.150–138.963 s; candidate range is 122.355–123.521 s. The slow baseline
observation is retained. All six outputs match all nine arrays bitwise; every
candidate selects 524 incoming and 524 outgoing Lt contractions without fallback.
Rust aggregate and per-run/pair gates pass, with a faster median and p95. This is
a loading improvement over the scoped-Lt route, not a new original-Torch baseline.
[Three-pair evidence](../provenance/protenix_base_buffered_receipts_three_pairs.json).

## CK block-scheduling screen

An isolated, source-hash-pinned CK header derivative changes only block-to-batch
scheduling (plus helper/class names), preserving the maintained 128/32/4 tile
arithmetic. Batch groups of 1, 16, 64 and 1000 cover all 32,000 tile/head/batch
combinations exactly once, including final partial groups. All eight random and
large-bias checks are bitwise equal to maintained CK. Median synthetic times are
20.369, 20.604, 20.489 and 20.315 ms respectively. The best difference is only
0.27%; no scheduling change is promoted or sent to a full-model speedup trial.
[Screen evidence](../provenance/protenix_base_ck_swizzle.json).

## Dual-projection reuse after buffered loading

Repeating the storage experiment with the buffered Rust bridge in both arms
changes the first matched result to **123.362992 -> 121.635853 s E2E**
(**1.01420x**, 1.40% less time). Model-call times are 98.966770 -> 97.410903 s.
All nine arrays remain bitwise equal and unchanged comparison gates pass.
The candidate uses one allocation for 2096 projections and 1048 consumed pairs.
This is one pair, not a repeated trend or a default-profile promotion. The earlier
negative result with unbuffered receipt parsing remains part of the record.
[Buffered reuse pair](../provenance/protenix_base_dual_reuse_buffered_first_pair.json).

## Static diffusion-bias audit

The global diffusion transformer has 24 attention blocks called over 200 steps.
An isolated audit preserves the original calculations and checks all 4776 later
bias tensors bitwise against the first 24. Pair-source identity/version and
projection parameters remain stable; all nine prediction arrays match the earlier
buffered route. The retained tensors occupy 1.536 GB. This establishes invariance
on the measured Base N1000 C10/D200/S1 workload; it is not a cache-mode speedup
claim. The subsequent cache-mode first pair is recorded below.
[Audit evidence](../provenance/protenix_base_diffusion_bias_audit.json).


The global cache-mode first pair measures **122.429573 -> 119.582363 s E2E**
(**1.02381x**, 2.33% less time), with model calls 98.352285 -> 95.130141 s.
It creates 24 biases and reuses them 4776 times across all 200 steps. All nine
arrays are bitwise equal and unchanged Rust comparison gates pass. This is one
pair, not a repeated trend; adaptive conditioning, attention and output gates
still execute on every step.
[First cache pair](../provenance/protenix_base_diffusion_bias_first_pair.json).

A separate audit of the six local atom-attention blocks also finds all 1194
later biases bitwise equal to the first six. Static pair/trunk inputs and
encoder/decoder parameter versions remain fixed; all nine predictions match
the global-cache anchor. Retained local biases occupy 98,304,000 bytes.
The first full local-cache pair measures **120.024906 -> 112.687053 s E2E**
(**1.06512x**, 6.11% less time), with global caching and buffered receipt reading
enabled in both arms. All nine arrays remain bitwise equal and unchanged Rust
comparison gates pass. This first pair has now been followed by two alternating repetitions.
[Local first pair](../provenance/protenix_base_local_bias_first_pair.json).
[Local audit](../provenance/protenix_base_local_bias_audit.json).


The complete **three BA/AB/BA pairs** measure E2E medians
**120.024906 -> 113.317563 s**, **1.05919x**, or **5.59% less time**.
All three paired ratios favor caching (1.05374x to 1.09290x); baseline range is
119.818–123.845 s and candidate range 112.687–113.708 s. All six outputs match
all nine arrays bitwise and unchanged Rust pair/aggregate gates pass. Published
kernels, global bias caching, scoped Lt, buffered bridge, FP32 N1000 C10/D200/S1,
seed101, HIP7 and unflushed caches are common to both arms. This remains an
increment over the global-cache route, not a new direct original-Torch ratio.
[Three-pair evidence](../provenance/protenix_base_local_bias_three_pairs.json).


### HIP Graph after global and local bias caching

The cache-aware denoiser trial preserves both bias caches and scoped triangle Lt.
One matched pair measures **113.316117 s off -> 113.569439 s on** complete E2E
(**0.99777x**, 0.22% more time). Model calls are 88.917637 -> 88.841963 s.
Capture costs 0.078879 s; all 200 denoising steps execute with one capture and
199 replays. All nine outputs match the same-worker graph-off control and the
previous eager-cache anchor bitwise; unchanged Rust comparison gates pass.

The experiment validates graph/cache composition but observes no useful E2E
benefit, so graph is not selected for the next combined optimization. Logical
steps and actual Python cache visits are checked separately; parameters, fixed
inputs and retained bias versions remain guarded across replay. Peak allocated
memory is 10,644,840,960 bytes. This single pair does not rule out other graph
partitions or persistent multi-request reuse.
[Cache-aware graph evidence](../provenance/protenix_base_cached_diffusion_graph.json).

### Dual-projection storage combined with both bias caches

On top of global/local bias caching, scoped Lt and buffered receipts, one
matched pair measures **113.479850 -> 112.678684 s E2E** (**1.00711x**, 0.71%
less time) when dual-projection storage reuse is enabled. Model calls are
88.864033 -> 88.112038 s. All nine arrays remain bitwise equal and unchanged
Rust comparison gates pass; the candidate records 2096 projections and 1048
consumed pairs. Graph remains off. This modest incremental result is one pair,
not a repeated combined trend or a direct comparison against original Torch.
[Combined first pair](../provenance/protenix_base_combined_reuse_first_pair.json).

### Diffusion profile after bias caching

A qualified event probe on the eager cache route preserves all nine arrays
bitwise. Across D200, inclusive denoiser time is 16.021 s, global transformer
7.770 s, atom encoder 3.048 s (its transformer 1.803 s), and atom decoder 1.681 s.
The encoder small MLP contributes only 0.309 s; its three static projections
contribute approximately 0.041 s combined. Further memoization of those small
modules is therefore deferred while global diffusion attention/transition costs
are examined. Nested event intervals include host launch gaps and must not be
summed into an E2E improvement claim.
[Qualified profile](../provenance/protenix_base_diffusion_static_profile.json).

### Global diffusion attention and transition diagnosis

A second qualified probe observes 4800 calls per global-block component; all
nine output arrays match the eager cached route bitwise. Inclusive diagnostic
totals include attention 6.661 s, conditioned transition 2.709 s, SDPA core
1.976 s, and adaptive normalization 0.861 s. Q/K/V projections total about
1.125 s. These nested intervals include extra instrumentation overhead and are
not directly additive or comparable as uninstrumented E2E measurements.
The SDPA input is FP32 `[1,16,1000,48]`, with strides `[768000,48,768,1]`.
[Global profile](../provenance/protenix_base_diffusion_global_profile.json).

### Diffusion transition elementwise fusion: provisional first pair

The published HIP SiLU/product and sigmoid/product kernels pass 16 exact
synthetic checks for the global/local transition shapes and ordinary/large
activations. An isolated adapter applies them to 24 global and six local
conditioned transitions, retaining the normalization and linear modules.
All 6000 transition calls are covered; all nine complete-model arrays remain
bitwise equal to the prior combined-reuse route.

One matched pair is **118.374007 -> 111.950568 s E2E** (1.05738x), with model
calls 93.792649 -> 87.935992 s. Unchanged Rust gates pass. However, this off-arm
baseline is noticeably above the previous approximately 113 s observations.
The observation is retained; the completed alternating repetitions are summarized
below. No default-profile promotion is implied. Both bias caches, dual storage reuse,
scoped Lt and buffered receipts are enabled in both arms; graph is off.
[Synthetic evidence](../provenance/protenix_base_diffusion_transition_synthetic.json),
[first paired evidence](../provenance/protenix_base_diffusion_transition_first_pair.json).


Across **three BA/AB/BA pairs**, transition fusion measures E2E medians
**112.610351 -> 112.040998 s**, **1.00508x**, or **0.51% less time**.
Baseline range is 112.483–118.374 s and candidate range 111.951–112.386 s;
the initial slow baseline is retained. All six outputs match all nine arrays
bitwise, every fused run covers 30 blocks times 200 steps, and unchanged Rust
pair/aggregate gates pass. The final pair improves only 0.086%, so this is a
small observed incremental gain, not evidence of a broad 5.7% improvement.
[Three transition pairs](../provenance/protenix_base_diffusion_transition_three_pairs.json).

### Attention gate/layout HIP screen

A new hand-written gfx90a kernel folds sigmoid gating and layout conversion into
the fresh gate-projection buffer. Four synthetic checks (two layouts, two input
ranges) are bitwise equal. Allocation-inclusive medians for the isolated gate
operation are 51.12 -> 39.60 microseconds for token-major attention output and
71.52 -> 41.12 microseconds for head-major output. Linear projections are excluded;
these are not E2E gains. A published, pinned bundle is used for the subsequent
full-model trial, whose result is recorded below.
[Gate screen](../provenance/protenix_base_diffusion_gate_screen.json).


The full-model gate trial covers all 4800 global attention calls and preserves
all nine arrays bitwise against the preceding transition-fused route. The
observed output layout is token-major in every call. Its matched E2E pair is
**112.048548 s off -> 112.477773 s on** (**0.99618x**, 0.38% more time).
Model calls are 88.035941 -> 87.814773 s, so the small forward reduction does
not become an E2E gain. Unchanged Rust comparison gates pass. This candidate
is retained as experimental evidence and is not selected for the maintained
fast path. Microbenchmark percentages do not represent complete inference gains.
[Gate first pair](../provenance/protenix_base_diffusion_gate_first_pair.json).

## Maintained N1000 optimization entry

The maintained package is `python/fireprot/adapters/protenix/base_n1000`.
`FIREPROT_ENABLE_BASE_N1000_OPTIMIZATIONS=1` requests its complete composition;
`0` (the default) preserves the existing worker path. The composition selects
scoped triangle Lt, dual-projection storage reuse, global/local diffusion bias
caches and HIP transition products. Graph and the rejected attention-gate kernel
are excluded. The context is selected through the Rust capability registry;
Python owns its tensor guards and temporary model hooks.

Use a fresh copy of `configs/requests/protenix-base-n1000-fireprot.json`, updating
`run_id` and `output_dir`, with the new DiamondHill profile:

```bash
scripts/infer.sh /tmp/my-base-request.json \
  configs/diamondhill/protenix-base-n1000-optimized.json
```

This is an explicit Base FP32 N1000 C10/D200/S1 route. Unsupported models,
workloads or token counts are rejected when it is enabled; use the ordinary
profile for other buckets. Atom count is read from input features, rather than
fixed to the 7979-atom performance example. Each component retains its coverage,
input/version, stream and buffer-lifetime checks, and is restored on failure.
`base-n1000-optimizations.json` records the chosen context and atom count;
component records retain the detailed counters.

The profile uses the existing published AMD/CK/dual bundles and a retained
buffered bridge at
`runtime/bridges/protenix-base-buffered-e8f166bb.so` (SHA256
`e8f166bb0a7f13f2600702207a7d07660f5193dbcf0de177fddd21bdfb51f965`).
This artifact is installed on DiamondHill. On another host, build the Rust bridge,
set `FIREPROT_BRIDGE_LIBRARY` to that build's actual path, and qualify the changed
artifact; do not treat a different build as this measured binary. Historical
root CLI/bridge artifacts and experimental workers are preserved.

The initial maintained-context runs match all nine arrays bitwise at both
7979 atoms (umem, 112.637 s) and 7711 atoms (5ZMW, 113.527 s). Those runs preceded
the final Rust-registry selection wiring. Its first run failed before model
execution because of an incorrect registry method name; the failure was retained,
the call corrected, and a real-registry test added. The corrected registry-mediated umem run now passes all nine arrays bitwise
at 112.085780 s and records the selected Rust implementation. The registry-mediated 5ZMW repeat also passes all nine arrays bitwise at
112.727518 s with 7711 atoms. The direct native E2E series is still pending. These timings
are migration checks, not a new original-Torch speedup claim.

[Promotion map](../provenance/protenix_base_promotion_map.json),
[umem pilot](../provenance/protenix_base_maintained_umem.json),
[5ZMW pilot](../provenance/protenix_base_maintained_5zmw.json).

[Registry-mediated umem qualification](../provenance/protenix_base_maintained_registry_umem.json).

[Registry-mediated 5ZMW qualification](../provenance/protenix_base_maintained_registry_5zmw.json).
