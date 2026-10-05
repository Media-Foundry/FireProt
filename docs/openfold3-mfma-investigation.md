# OpenFold 3 FP32 MFMA investigation

Status: read-only ISA and historical-profile review, 2026-10-06. No new GEMM
implementation, disassembly result, numerical equivalence or speedup is claimed.
The conditioning-cache paired benchmark has now completed successfully. A
bounded QK/PV observation in a separate copied worker now passes all 11 model
arrays exactly and records CPU/GPU correlations. Disassembled ISA remains pending.

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

## Next evidence gates

1. After the current paired sequence, capture actual QK/PV tensor shapes,
   strides, dtypes and kernel correlations on the retained long-input workloads.
   Separate attention GEMMs from projections and triangle multiplication. Check
   that this diagnostic preserves all 11 model output arrays.
2. Locate only the corresponding installed library code objects. Work on copies;
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
