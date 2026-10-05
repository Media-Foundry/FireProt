# OpenFold 3 FP32 MFMA investigation

Status: bounded native QK/PV observation and matched code-object review complete,
2026-10-06. Both selected native kernels contain FP32 16x16x4 MFMA instructions.
No replacement GEMM, exact reproduction of its accumulation order or GEMM speedup
is established. The bounded observation preserves all 11 model arrays; the
read-only disassembly and remaining fusion gates are recorded below.

## Source and verified ISA facts

The user supplied `docs/isa/CDNA2.pdf`, titled *AMD Instinct MI200 Instruction
Set Architecture*. Its SHA-256 is
`6c8b6e86fddb489437765c1d4c290c223e7668576bbd2ea9c2ebb4d45e3e2212`.
The PDF remains untracked. The authorized GPT-6-luna xhigh reviewer supplied
the follow-up review; the instruction table and relevant matrix/denormal text
were independently read from the local PDF.

| Subject | PDF page / printed page | What the reviewed passage establishes |
| --- | --- | --- |
| MFMA register files | 53-54 / 45-46 | Matrix C/D use accumulation registers; `V_ACCVGPR_READ/WRITE` move values between accumulation and architectural registers. |
| FP32 denormals | 54 / 46 | FP32-input MFMA supports allowing/flushing denormals under `MODE.denorm`. This differs from the documented F16-input behavior. |
| Dependency waits | 55-57 / 47-49 | Same-opcode SrcC forwarding and other operand/consumer hazards have distinct wait requirements. A universal zero-wait assumption is invalid. |
| `V_MFMA_F32_16X16X4F32` | 156 / 148 | `16x4` times `4x16` plus C; one block, eight passes; the opcode table lists one Arch VGPR for each A/B operand and four Acc VGPRs for C/D. |
| `V_MFMA_F32_32X32X2F32` | 156 / 148 | `32x2` times `2x32` plus C; one block, sixteen passes; the table lists one Arch VGPR for each A/B operand and sixteen Acc VGPRs for C/D. |

For a dot-product dimension of 32, dividing by the opcode K dimensions gives
eight or sixteen K fragments respectively. This is arithmetic about the named
instructions, not evidence of the actual hipBLAS implementation or its latency.
Instruction "passes" must not be treated as a complete kernel cost model.

The reviewed passages do not establish a complete lane/register-to-element
mapping for a proposed fused kernel, the exact native library accumulation
sequence, or bitwise equivalence to Torch GEMM. Matching a matrix shape or final
dtype is insufficient. Full QK-softmax-PV fusion also requires a layout suitable
for row reductions and a PV accumulation sequence; changing to an online softmax
algorithm can change arithmetic order.

## What the existing trace adds

`provenance/openfold3_historical_gemm_symbols.json` extracts twelve GEMM symbol
records from the SHA-verified, pre-optimization N1536 native trunk diagnostic.
The largest recorded entries include these prefixes:

| Symbol prefix | Recorded self-device time |
| --- | ---: |
| `Cijk_Ailk_Bljk_SB_MT32x64x32_MI16x16x4x1_SE` | 54.167 s |
| `Cijk_Ailk_Bljk_SB_MT64x32x32_MI16x16x4x1_SN` | 35.418 s |
| `Cijk_Alik_Bljk_SB_MT64x64x32_MI16x16x4x1_SN` | 24.755 s |

These are historical profiler attributions, not clean latency measurements or
current optimized-path bottlenecks. The original run reported duplicate
ROCTracer flows. Do not sum these values with nested ATen/stage totals, infer
exact launch counts, or claim they represent QK/PV alone. Their input-shape lists
are empty and this aggregate does not provide the necessary correlation.

The literal `MI16x16x4x1` and `MI32x32x2x1` tokens make an existing native MFMA
implementation a plausible hypothesis. They do not prove the actual opcodes.
The next experiment must therefore test a specific reduction in work or data
movement; merely replacing a library call with an MFMA intrinsic is not an
established optimization.

## Accepted bounded N1536 observation

`provenance/openfold3_attention_trace_n1536.json` records the corrected profiler
with all 11 original-native arrays exact. It observes one B32/H4/N1536/D32 call,
which can belong to chunk tuning. It is not a whole-model ranking or clean
latency measurement. The first observer failed under inference mode and was
excluded; that failed trial was not silently accepted or used for speedup.

| Stage | Actual BMM input shapes | Input strides | Correlated library prefix |
| --- | --- | --- | --- |
| QK | `[128,1536,32]`, `[128,32,1536]` | `[49152,32,1]`, `[49152,1536,1]` | `Cijk_Ailk_Bljk_SB_MT64x32x32_MI16x16x4x1_SN` |
| PV | `[128,1536,1536]`, `[128,1536,32]` | `[2359296,1536,1]`, `[49152,32,1]` | `Cijk_Ailk_Bljk_SB_MT32x64x32_MI16x16x4x1_SE` |

Original query/key/value strides are `[196608,32,128,1]`; the trace records
cloning/copying into the BMM layouts. Each BMM links through an external ID to
one unique `hipExtModuleLaunchKernel` correlation and its GPU kernel. Complete
symbols and raw-trace hashes are preserved in the evidence. This closes the
shape/correlation gap for this selected call, but does not prove the named ISA
instructions, accumulation sequence or kernel choice at other chunk sizes.

## Matched native code objects

The selected trace symbols now match the installed rocBLAS gfx90a code object,
AMD metadata and descriptor entries. A read-only review worked on SHA-verified
copies. The primary agent independently matched the full disassembly symbols,
counted MFMA opcodes and decoded each ELF64 kernel descriptor's entry offset and
LDS size. Evidence and archived code objects/excerpts:
`provenance/openfold3_native_mfma_review.json`.

| Stage | Descriptor / entry | Actual opcode | LDS bytes | VGPR / SGPR |
| --- | --- | --- | ---: | ---: |
| QK | `0x1d29c0` / `0x414f00` | `v_mfma_f32_16x16x4f32` | 12,800 | 88 / 63 |
| PV | `0x1d3180` / `0x44f200` | `v_mfma_f32_16x16x4f32` | 29,184 | 248 / 58 |

Each selected static function body contains 80 of these instructions and no
32x32x2 FP32 MFMA. Static counts are not executed counts. Both selected solutions
use DepthU=32, GSU=1, StreamK=0 and no global accumulation. **QK reduces K=32;
PV reduces K=1536**, respectively one and 48 DepthU tiles. The review summary
initially conflated these dimensions; the checked-in evidence corrects that.
There is no split-K/Stream-K cross-CTA accumulation in either selected solution.
Neither exact scalar accumulation order nor occupancy follows from these facts.

A second PV body with GLVWA1/GRVW1 is excluded: the trace selects GLVWA2/GRVW2.
The complete symbol names retain all tokens, including LBSPPB128/LPA0/WSGRB1.
Torch's reported HIP version and the resolved `core-10.0` install directory are
recorded separately; the directory name alone establishes no version conflict.

The row-major-to-column-major adapter suggests kernel A=key and B=query for QK,
so query rows would occupy the MFMA N axis. This is an inference: the trace does
not record runtime kernarg pointer identities. A score-only probe must resolve
this and prove bitwise equality before fusing softmax. Four/eight query rows
across 1536 keys need 24/48 KiB of score LDS, but waste 75%/50% of the MFMA N=16
fragment. Reusing staging LDS requires proven lifetimes/barriers; resource fit
alone does not establish a speedup.

## Packed QK reduction reproduced for the traced shape

The initial contiguous-K4 prototype passes integer lane calibration but fails
full random scores. Sparse-K and cancellation diagnostics isolate an ordering
issue: adjacent cancellation gives native2.0 versus candidate1.0, while single-K
coverage passes. The authorized ISA review narrows the native K32 path to eight
MFMA issues per accumulator chain, with LDS/register patterns suggesting even
then odd K entries within each K8. Static offsets alone do not prove all lanes.

An input-order screen confirms that hypothesis on retained16x16 native-reference
tiles. The corrected HIP prototype implements
`k=(fragment/2)*8+2*k_lane+(fragment%2)` and passes all packed
`[128,1536,32] @ [128,32,1536]` scores bitwise for random, cancellation, subnormal,
signed-zero and nonfinite inputs. Its native reference selects exactly the
previously traced QK kernel. Source, traces, rejected attempts and the corrected
artifact are retained in `provenance/openfold3_mfma_qk_v2_probe.json` and the
[probe notes](../experiments/mfma_qk/README.md).

This is not model integration, original-stride handling, proof for other chunk
sizes or a speedup. The one-wave probe still materializes scores globally.
The next fusion must preserve this validated K grouping and the existing softmax
evaluation order while including packing/allocation costs in comparisons.

## Next evidence gates

1. Completed for the bounded B32/N1536 call: capture actual QK/PV tensor shapes,
   strides, dtypes and kernel correlations on the retained long-input workloads.
   Separate attention GEMMs from projections and triangle multiplication. Check
   that this diagnostic preserves all 11 model output arrays.
2. Matched code objects are now archived; remaining work is exact runtime K/lane
   mapping. Locate only corresponding library code objects and work on copies;
   never pass a published input ELF to an extraction command that can rewrite it.
   Record source/copy hashes and inspect opcode choice, operand/accumulator
   layout, K-fragment ordering, split-K, output conversion, floating-point modes
   and dependency scheduling. Keep heavy library scans outside E2E measurement.
3. If the native sequence can be reproduced, isolate a QK tile experiment with
   the current softmax and native PV. Validate real-layout random/adversarial
   inputs and the exact-model gate before extending the fusion. A numerical
   mismatch rejects the candidate; the existing gate is not relaxed.
4. Compare against the actual selected library kernel including packing,
   allocation and conversion overhead. Only a passing complete-model comparison
   can justify paired E2E measurement and eventual dispatch promotion.

If equivalent arithmetic cannot be established, retain native GEMM and target
measured copies, layout conversion or launch overhead around it. This is a
decision contingent on evidence, not a claim that further GEMM gains are flat.
