# Packed QK MFMA probe

Status: the corrected K8 even/odd grouping passes integer lane calibration and
all five packed B128/N1536/K32 score cases bitwise. Full attention/model integration
and performance remain untested. The earlier contiguous-K4 candidate is rejected.
Do not build or run this probe on DiamondHill while the sequential E2E benchmark
is active. Python syntax checking is not a HIP compilation or GPU acceptance test.

For the earlier candidate, the local HIP 7.14 compiler produced a gfx90a library.
Inspection of its
extracted code object finds eight static `v_mfma_f32_16x16x4f32` instructions,
wave64, 26 VGPRs, 24 SGPRs, no reported spills, and no fixed LDS/private segment.
This is the one-wave diagnostic, not the proposed fused softmax kernel.
The direct local compiler check is separate from the required Rust build and
pinned runtime on DiamondHill; its artifact is not published or integrated.
See [compiler evidence](../../provenance/openfold3_mfma_qk_local_compile.json).

The accepted native B32/H4/N1536/D32 trace packs QK into row-major inputs
`[128,1536,32]` / strides `[49152,32,1]` and `[128,32,1536]` / strides
`[49152,1536,1]`. Its selected native code object uses FP32 16x16x4 MFMA.
See `docs/openfold3-mfma-investigation.md` and the pinned disassembly evidence.

`qk_probe.cpp` tests a single-wave 16x16 output tile with A=packed key, B=query,
lane coordinates `lane%16` and `lane/16`. Full K32 uses
`k=(fragment/2)*8+2*k_lane+(fragment%2)`: even entries then odd entries within
each K8 group. Raw lane calibration uses contiguous K0..3 independently.
The CDNA2 manual does not establish these lane mappings; acceptance comes from
the bounded GPU comparisons, supported by the native LDS/issue-order review.
The native QK has K32; native PV has K1536 and is outside this probe.

The output hypothesis is query=`lane%16`, key=`4*(lane/16)+register` within
each tile. This follows the operand/output orientation used by the repository's
existing `dual_projection_kernel` and `qkv_detail::project_qkv` MFMA code. It
replaces an initially proposed transpose. The chosen mapping passes the raw
integer calibration, which remains mandatory and does not alone prove K32 order.

The first GPU gate executes one K4 instruction with exact small-integer encoding
and writes each lane's four raw accumulator values. `probe.py` rejects the
hypothesized layout if those bits do not match. Only then does it compare full
packed K32 scores against native `torch.bmm`, including random, cancellation,
subnormal, signed-zero and nonfinite inputs under strict FP32 inference. It
stops on the first mismatch and records failure; no tolerance or fallback hides
it. Packing and model attention are not replaced. This packed score experiment
does not establish behavior for original strided QKV or other GEMM signatures.
The random reference call retains a Kineto trace and requires one BMM with the
expected packed layouts and one uniquely correlated GEMM kernel. A first-use
runtime initialization memset is retained separately; unknown additional kernels
are rejected. The probe explicitly selects Torch's `cublas` preference on ROCm
(hipBLAS), matching the pinned OpenFold3 entry-point configuration. It reuses
`experiments/attention_trace/probe.py`; include that source when copying this
experiment. Compare its actual symbol with the accepted model trace before
attributing any score agreement to reproduction of that particular native kernel.
Profiler timings are not performance evidence.

## First GPU results: rejected contiguous-K4 candidate

The Rust-built experimental DiamondHill artifact passes the single-instruction
integer lane calibration. The first probe then fails its native-kernel observation
gate: standalone Torch defaults to hipBLASLt, unlike the model entry point's
explicit hipBLAS preference, and the trace also includes an initialization memset.
That failed run remains retained and is not a numerical acceptance result.

The corrected probe selects hipBLAS and records initialization separately from
one correlated GEMM. Its full native kernel name exactly matches the accepted
model QK trace. Random K32 inputs differ in193,776,224 of301,989,888 score elements.
The probe stops; cancellation, subnormal, signed-zero and nonfinite cases have
not run. Lane calibration alone does not prove the full reduction order or
arithmetic equivalence. Investigate fragment order and instruction semantics
before any softmax fusion or model integration; do not loosen the bitwise gate.

Both attempts, source/build/publication pins and the result archive are recorded
in [GPU evidence](../../provenance/openfold3_mfma_qk_probe.json). These are
concurrent numerical diagnostics, not timing measurements.

## K grouping correction and full score acceptance

`diagnose.py` retains the full native B128/N1536 GEMM shape, then compares slices
with candidate16x16 tiles. Single-K checks cover all32 K positions and pass.
Dense random tiles differ by up to2.98e-8. Adjacent cancellation produces native
2.0 versus candidate1.0 (absolute error1.0, not one ULP), exposing a reduction
ordering difference without changing the exact gate.

`reorder_screen.py` checks explicit input-order hypotheses against those pinned
reference tiles. Only even/odd grouping within each K8 passes all five tile
cases; contiguous K4, middle-K4 swap and stride8 still fail. The ISA review's
LDS/source-register order supports that mapping, but is not an independent proof
of every lane/wave mapping or internal floating-point addition sequence.

The corrected source implements that grouping directly in its K address, without
runtime input reordering. A fresh Rust-built/published artifact then passes all
301,989,888 scores in each of random, cancellation, subnormal, signed-zero and
nonfinite cases, plus raw lane calibration. The observed native GEMM name exactly
matches the accepted model QK trace. Evidence and archived diagnostics:
[v2 score acceptance](../../provenance/openfold3_mfma_qk_v2_probe.json).

This establishes these five packed B128/N1536/K32 comparisons, not other batch
sizes, original QKV strides, arbitrary inputs, complete attention or model parity.
The probe is not tuned and no speedup is claimed.

## Experiment gates

1. Require any active E2E sequence to be terminal before new GPU diagnostics or
   heavy DiamondHill build/inventory work. The first three-pair combined sequence
   completed and was archived before these probes started.
2. Build `qk_probe.cpp` via `build.py`, which uses the existing Rust HIP build/cache
   service. Use the pinned OpenFold3 environment and an isolated cache. Retain
   the full compiler command, source/header receipts and diagnostics; inspect
   generated ISA and floating-point modes before interpreting numerical results.
3. Publish the C ABI artifact with Rust's ordinary schema-2 artifact workflow.
   Set `FIREPROT_QK_MFMA16_BUNDLE` and `FIREPROT_QK_MFMA16_BUNDLE_SHA256` to that
   immutable bundle, then run `probe.py NEW_DESTINATION` on one available GCD.
   The package is `fireprot_experimental_qk_mfma16`; C ABI revision is
   `experimental-qk-mfma16-f32-v1`.
4. A lane failure requires calibration of the mapping; a K32 failure requires
   investigation of accumulation order, operand orientation or floating-point
   modes. The gate remains bitwise. Record the actually selected native kernel
   before claiming to reproduce the previously traced native implementation.
5. Only a passing score experiment can support a subsequent LDS softmax fusion
   trial. That still requires operator and complete-model tests before E2E.

The one-wave diagnostic is not tuned. It retains a full global score output and
offers no memory-traffic saving by itself. A separate QK-plus-softmax prototype
now compiles locally; its GPU correctness and performance remain untested.

## Fusion prototype after score acceptance

The [local prototype](FUSION.md) covers16 query rows across16 waves (1024 threads).
Each wave computes six16-key tiles at N1536, transposing each band immediately
before computing the next. This avoids retaining24 old MFMA accumulator values
alongside24 final softmax values per thread. The complete row values occupy
96 KiB across registers. Six staged transposes use a16-by-256 FP32 LDS tile
(16 KiB) with a column XOR swizzle. After each transpose, each wave
owns one query row; lane `l` receives columns `l + 64*j`, matching the existing
softmax's row layout and local evaluation order. No online-softmax rescaling is
introduced. The block would write probabilities for the original native PV.

This implementation remains GPU-untested. It avoids intentionally
discarding most columns of a16-wide MFMA fragment, but large-block register
pressure, accumulator lifetime, LDS bank conflicts, barriers, spills and
residency still need runtime validation and measurement. The transpose must preserve
every FP32 bit. Output allocation/filling, original QKV packing, model chunk
selection and the native PV cost must remain included when comparing paths.
