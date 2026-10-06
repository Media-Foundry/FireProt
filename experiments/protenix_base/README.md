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
