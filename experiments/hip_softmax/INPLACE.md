# Pending experiment: owned score buffer reuse

Status: source preparation only. No GPU numerical or timing result yet.
The active three-pair N1536 experiment uses the original out-of-place boundary.
Do not change its copied workers or immutable bundle.

The current kernel loads each score row into lane-local values before any output
store. Its wave64 reductions keep all row lanes participating. Independent rows
are mapped either directly or through a bijection over groups of eight outer
rows. No row consumes another score row, and the C++ pointers are not `restrict`.
These source properties support testing identical score/output pointers when
the score allocation is privately owned and disjoint from both biases.

`screen_inplace.py` calls the existing pinned C ABI with a separately allocated
work buffer. Both mask and pair bias use independent allocations. It compares
FP32 output bits against native mask-add, bias-add and softmax at N1000/B256 and
N1536/B128, with both bias layouts and zero/partial/nonzero/special-value cases.
It then alternates the selected packed HIP out-of-place and in-place variants,
including bias packing and output allocation but excluding score reset and QK/PV.
Five observations follow two warmups; peak additional Torch allocations are
recorded separately from latency.

The N1536/B128 FP32 score contains 1,207,959,552 elements (4.5 GiB); N1000/B256
contains 1,024,000,000 elements (about 3.815 GiB). Avoiding a second allocation of
this size is a theoretical workspace saving, not a measured whole-model peak
reduction. Score reads and probability writes still occur, so lower memory
traffic or latency must not be assumed.

After the current E2E experiment finishes, run this screen through the same
pinned published loader environment on an explicitly selected idle GPU. A later
worker trial must ensure QK scores have no other consumers and reject aliases
with mask/bias. Full-model validation must record effective chunk sizes: the
memory reduction can alter upstream autotuning, GEMM shapes and numerical
results. Require all 11 arrays bitwise equal at both lengths before any paired
E2E or default-dispatch change. This script adds no maintained in-place API.
