# Protenix Base experiments

These are opt-in experiments, not production defaults. Rust `measure-inference`
owns model invocation, device isolation and receipts. Use HIP devices 6/7 only.
The current workload is Base-default v0.5.0, FP32, N1000, C10/D200/S1,
seed 101, strict determinism and uninitialized-memory filling.

## Module profiling

`stage_probe.py` installs temporary module hooks in a frozen worker snapshot.
It reports inclusive GPU event intervals, including host launch gaps. Parent
and child rows overlap and must not be summed as disjoint kernel time.
Instrumentation changes timing; its invocation is not an E2E speedup result.
Compare all output arrays with the uninstrumented route to check probe effects.

## Pairformer HIP Graph experiment

`recycle_graph.py` is installed only around the actual model invocation in a
frozen worker. `FIREPROT_BASE_RECYCLE_GRAPH=off/on` selects the control/treatment.
On ROCm, the PyTorch `torch.cuda.CUDAGraph` API uses the HIP backend.

- Capture only the trunk's 48-block Pairformer stack; sampling and feature
  preparation remain outside capture.
- Warm the capture stream with the first actual recycle. Capture and replay
  the second actual recycle, then replay the remaining eight. Keep C10.
- Copy each cycle's fresh single/pair inputs into retained static buffers.
  Inputs, parameters and intermediates must remain valid for graph replay.
- Require fixed shape/strides, FP32, inference, unmasked pair inputs and the
  existing FireProt implementations. Unexpected signatures abort the trial.
- Check for unexpected RNG changes at capture and verify complete outputs.
  Capture failures stop the experiment; no silent eager fallback hides them.
- Include capture, static-buffer copies and graph setup in complete invocation
  timing. A faster isolated replay is insufficient for promotion.

The current numerical diagnostic compensates rigid translation/rotation and
reports all-atom aligned RMSD against the native prediction (user limit 0.1 Å),
relative CA pair-distance changes and separate confidence deltas. Prediction
agreement is not experimental structure accuracy. Generic historical CLI gates
remain unchanged. See `docs/protenix-base.md` for completed experiments.

`model_softmax.py` is an undeployed alternative prototype. Only the separate
`bias_softmax.cpp` synthetic operator screen has passed; this does not establish
trained-call or model qualification of that adapter.

## Denoiser HIP Graph experiment

`diffusion_graph.py` uses `FIREPROT_BASE_DIFFUSION_GRAPH=off/on`. Only the
neural denoiser is captured: the 200-step noise schedule, random augmentation,
noise generation and Euler updates remain eager and retain their original RNG
sequence. Step 1 warms the side stream, step 2 captures and replays, and the
remaining 198 steps replay. Each replay refreshes noisy coordinates and noise
level; constants retain their original identities and must not mutate. The
initial signature is deliberately limited to this N1000/7979-atom S1 case.
This experiment does not imply support for arbitrary proteins or sample counts.

## CK attention tile rescreen

`screen_ck_tiles.py OUTPUT [SOURCE_NAME TILE_LABELS_JSON]` builds a standalone
library through the Rust HIP build service. The exact N1000/head-last FP32 ABI
in `ck_tiles.cpp` has four variants; `ck_tiles_wide.cpp` contains the larger-tile
round. Only HIP6/7 are admitted by the screen. Seed101, strict determinism and
memory filling are enabled. Random and large-bias outputs are checked against
the maintained vector-bias CK operator before timing. Four warmup rounds precede
16 measured rounds, with alternating variant order. Output allocation/filling
is included; the bias is already contiguous. These are synthetic operator screens,
not Base structural accuracy or E2E evidence. Raw Rust build receipts are retained
beside `screen.json` in the experiment roots recorded in provenance.

## Padded N1000 hipBLASLt contraction screen

`screen_lt_padded.py OUTPUT` builds `lt_padded.cpp` through the Rust HIP build
service. The input is FP32 `[128,1000,1000]` with strides `[1000128,1000,1]`,
matching the current tiled dual projection; output is contiguous. Incoming and
outgoing contractions use separate Lt descriptors and up to 16 returned heuristic
algorithms. An external 32 MiB workspace is retained across calls. Comparisons
use Torch hipBLAS with TF32 disabled, seed101, strict determinism and memory
filling. Each algorithm must pass 2e-5 absolute/relative synthetic agreement
before timing. Output allocation/filling is timed; workspace creation is not.
This screen is not connected to model dispatch. A winner still requires trained
model correctness and complete invocation comparisons including setup costs.

## Lt setup diagnosis and Torch routing

`packed_probe.py` records first-call and aggregate event/wall intervals around
trained contractions. Its instrumented outputs must match the corresponding
uninstrumented route; its timings are diagnostic, not E2E benchmark replacements.
The direct Lt adapter's 2.11 s setup and first-call costs offset its contraction
gains in the first full-model pair.

`screen_torch_blas.py` compares Torch `hipblas`/`hipblaslt` preferences with the
same padded input strides. `model_torch_lt.py` scopes that preference to each
matching BMM and restores it in `finally`. This avoids an extra native extension
load and preserves all other calls' BLAS preference. It is intended only for the
single-threaded inference worker; it does not establish safety for concurrent
model threads sharing a process.

`model_global_blas.py` is a separate full-forward preference experiment. It
records requested/effective/restored settings; a preference is not proof that
every internal call selected that library. Both model experiments use frozen
workers and require full coordinate/relative-geometry checks and E2E comparison.
Neither changes the maintained worker's BLAS default.

`prepare_experimental_case.py` extracts the pinned 5ZMW chain-A sequence and
experimental CA coordinates. The generated development case is under
`benchmarks/protenix/experimental/5zmw_A`; metadata records source hashes and
residue mapping. Use the Rust diagnostic after inference:

```bash
cargo run --release -p fireprot-validation --example experimental_ca -- \
  benchmarks/protenix/experimental/5zmw_A sample-0.npz prediction.cif result.json A
```

This reports experimental CA RMSD/lDDT and verifies sequence/CIF/NPZ binding.
It does not reinterpret native-model numerical parity as experimental accuracy.

`triangle_probe.py` separates inclusive GPU-event intervals for triangle input
normalization, dual projection, contraction, packed normalization, output linear,
gate linear, and residual fusion. It eagerly loads the selected AMD extension
and adds hooks/wrappers, so its E2E is not performance evidence. All nine output
arrays must match the uninstrumented scoped-Lt anchor before interpreting timings.

## Dual-projection storage and tile experiments

- `dual_tiles.cpp` / `screen_dual_tiles.py`: seven FP32 MFMA tile shapes, exact
  comparison against the published maintained projection, alternating timing order.
- `quad_projection.cpp` / `screen_quad_projection.py`: five fused A/B kernels;
  no useful gain found despite exact synthetic outputs.
- `screen_dual_reuse.py`: retained output versus fresh allocation, deterministic
  fill enabled, changing-input overwrite checks. Initial allocation is excluded.
- `model_dual_reuse.py`: isolated model trial with two retained slots and guarded
  projection-to-BMM consumption on one stream. Requires 2096 projections and
  1048 consumed pairs. Full invocation timing includes initialization/loading.

Microbenchmark speedups are not model E2E results. The published C ABI kernel is
selected and authenticated by Rust; these adapters do not compile during inference.

`prepare_ck_swizzle.py` generates a separate MIT CK-header derivative from one
pinned installed header. It retains AMD notices and changes helper/class names
and block-index mapping. `ck_swizzle.cpp` / `screen_ck_swizzle.py` screen batch
scheduling groups without modifying installed CK headers or inner tile math.
Generate `fmha_swizzle.hpp` beside the source before building; the screen adds
that directory explicitly to compiler include paths and fingerprints both the
header and its provenance. All eight exact checks pass, but no useful gain was
measured, so the maintained schedule stays unchanged.

`diffusion_bias_cache.py` audits or reuses the 24 global diffusion attention biases.
Audit mode retains every original projection and checks the 4776 later biases
against the first 24 bitwise. Pair-source and projection-parameter identity/version
checks remain active. Cache mode reuses only those biases; adaptive single
conditioning, attention and output gates still run every step. The experiment
requires Base N1000, 200 steps and one sample, retaining 1.536 GB of bias tensors.
Its original-computation audit passes with all nine prediction arrays unchanged;
the first cache-mode full-model pair also preserves all nine arrays and reduces
E2E from 122.430 to 119.582 s. Repeated timing is still required before a stable
performance claim.

`diffusion_local_bias_cache.py` independently audits/reuses six local atom biases.
The audit retains original projections, checks 1194 later values bitwise, and
verifies static `p_lm`, `c_l`, `s_trunk`, parameter versions and the execution stream.
It runs on top of global bias caching, with its own mode and output record.
The audit and first cache-mode pair pass: all nine arrays are bitwise equal,
unchanged Rust gates pass, and complete invocation is 120.025 -> 112.687 s.
Three BA/AB/BA pairs now pass: E2E medians 120.025 -> 113.318 s (1.05919x),
all six outputs bitwise equal and unchanged Rust gates passing.

The existing graph experiments cannot be stacked with these wrappers unchanged:
scoped Lt rejects capture, and bias coverage counters count Python executions
rather than graph replays. A combined experiment must preserve the selected BLAS
route, validate static-cache lifetimes/streams across capture, and distinguish
logical model steps from eager/captured Python calls. Do not disable these checks
just to make replay succeed.

### Cache-aware denoiser graph trial

The optional `CacheReplayContract` in `diffusion_graph.py` supports a new isolated
combination with both bias caches. Create one contract, install global and local
cache contexts with it, then install the graph context innermost. The local hook
must bind the original denoiser signature before the graph wrapper is installed.
The graph checks parameter identities/versions, constant inputs and retained bias
versions before replay. The local denoiser hook still observes all 200 calls.
Global conditioning and global/local attention execute in Python twice (warmup
and capture), while the graph records one capture and 199 replays. Completion
requires both logical-step and Python-execution counts to match. Off mode keeps
all 200 Python executions and the original cache coverage requirements.

This captures only the denoiser; triangle Lt dispatch remains eager and retains
its 524 incoming/524 outgoing coverage gate. The prepared frozen worker is not a
maintained deployment profile. CPU contract checks and full-model graph-on/off correctness pass: all nine
arrays are bitwise equal, including against the earlier eager-cache anchor.
The matched E2E pair is 113.316 s off versus 113.569 s on (0.99777x).
Model time is 88.918 versus 88.842 s. No useful gain was observed; graph remains
disabled. Capture costs 0.0789 s, with one capture and 199 replays.

`diffusion_static_probe.py` measures the denoiser, its conditioning/global
transformer, atom encoder/decoder, and selected encoder projections/small MLP
after bias caching. It requires 200 calls per selected module and checks for
unclosed event intervals. Timing is inclusive, nested and instrumented. Require
all nine output arrays to match the uninstrumented cache route before using the
rows to choose further optimization targets.

`diffusion_global_probe.py` drills into 24 cached global diffusion blocks:
attention, adaptive normalization, Q/K/V/output/gate projections, conditioned
transition, and the SDPA core at `[1,16,1000,48]`. It requires 4800 observations
per group and restores hooks/SDPA dispatch on exit. Qualification against the
uninstrumented cache output is required before interpreting its diagnostic rows.
