# Packed QK MFMA probe

Status: local HIP compilation passed; GPU execution remains untested.
No lane mapping, numerical equivalence, latency gain or model integration is claimed.
Do not build or run this probe on DiamondHill while the sequential E2E benchmark
is active. Python syntax checking is not a HIP compilation or GPU acceptance test.

The local HIP 7.14 compiler produced a gfx90a shared library. Inspection of its
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

`qk_probe.cpp` tests a single-wave 16x16 output tile. It hypothesizes kernel
A=packed key, B=query, lane coordinates `lane%16` and `lane/16`, and eight
ascending K4 fragments. The CDNA2 manual does not establish these lane mappings,
and the native LDS swizzle/control flow does not by itself prove this K order.
The native QK has K32; native PV has K1536 and is outside this probe.

The output hypothesis is query=`lane%16`, key=`4*(lane/16)+register` within
each tile. This follows the operand/output orientation used by the repository's
existing `dual_projection_kernel` and `qkv_detail::project_qkv` MFMA code. It
replaces an initially proposed transpose of that mapping; neither proposal has
been executed in this new probe. The raw integer calibration remains mandatory.

The first GPU gate executes one K4 instruction with exact small-integer encoding
and writes each lane's four raw accumulator values. `probe.py` rejects the
hypothesized layout if those bits do not match. Only then does it compare full
packed K32 scores against native `torch.bmm`, including random, cancellation,
subnormal, signed-zero and nonfinite inputs under strict FP32 inference. It
stops on the first mismatch and records failure; no tolerance or fallback hides
it. Packing and model attention are not replaced. This packed score experiment
does not establish behavior for original strided QKV or other GEMM signatures.
The random reference call retains a Kineto trace and requires one BMM with the
expected packed layouts and one uniquely correlated GPU kernel. It reuses
`experiments/attention_trace/probe.py`; include that source when copying this
experiment. Compare its actual symbol with the accepted model trace before
attributing any score agreement to reproduction of that particular native kernel.
Profiler timings are not performance evidence.

## Future execution gates

1. Finish the active E2E sequence and verify it is terminal before any new GPU
   probe or heavy DiamondHill build/inventory work.
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
offers no memory-traffic saving by itself. QK-plus-softmax fusion remains future
work, with explicit LDS/register/occupancy and arithmetic-order constraints.

## Candidate fusion layout after score acceptance

A layout worth testing retains 16 query rows across 16 waves (1,024 threads).
Each wave computes six 16-key tiles at N1536, retaining 24 scores per thread.
The complete 16-by-1536 score tile is 96 KiB, distributed over registers rather
than allocated as one LDS array. Six staged transposes can use a 16-by-256 FP32
LDS tile (16 KiB before any padding/swizzle). After each transpose, each wave
owns one query row; lane `l` receives columns `l + 64*j`, matching the existing
softmax's row layout and local evaluation order. No online-softmax rescaling is
introduced. The block would write probabilities for the original native PV.

This is a design hypothesis, not implemented code. It avoids intentionally
discarding most columns of a 16-wide MFMA fragment, but large-block register
pressure, accumulator lifetime, LDS bank conflicts, barriers, spills and
residency still need disassembly and measurement. The transpose must preserve
every FP32 bit. Output allocation/filling, original QKV packing, model chunk
selection and the native PV cost must remain included when comparing paths.
