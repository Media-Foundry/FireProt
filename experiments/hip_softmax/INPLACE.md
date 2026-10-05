# Pending experiment: owned score buffer reuse

Status: GPU operator screen complete; no model/E2E validation or integration.
The completed three-pair N1536 experiment uses the original out-of-place boundary.
Keep its copied workers and immutable bundle intact.

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

The screen ran after the E2E experiment finished, through the same pinned
published loader on HIP5/PCIb3/SMI7. All 16 comparisons pass bitwise. Five
observations follow two warmups, with alternating variant order:

| Shape/layout | Out-of-place median | In-place median | Saved extra allocation |
| --- | ---: | ---: | ---: |
| N1000/B256 starting | 10.970 ms | 11.095 ms | 4,096,000,000 bytes |
| N1000/B256 ending | 10.964 ms | 11.071 ms | 4,096,000,000 bytes |
| N1536/B128 starting | 9.882 ms | 9.863 ms | 4,831,838,208 bytes |
| N1536/B128 ending | 10.268 ms | 10.256 ms | 4,831,838,208 bytes |

These observations establish operator allocation savings, not a latency win.
N1000 is about 1% slower; the N1536 differences are only 0.1-0.2%. Keep the
out-of-place path for maintained integration. Full observations, source/library
pins and limits are in `provenance/openfold3_hip_softmax_inplace_screen.json`.
Isolation checks were point-in-time, not continuous monitoring.

A later
worker trial must ensure QK scores have no other consumers and reject aliases
with mask/bias. Full-model validation must record effective chunk sizes: the
memory reduction can alter upstream autotuning, GEMM shapes and numerical
results. Require all 11 arrays bitwise equal at both lengths before any paired
E2E or default-dispatch change. This script adds no maintained in-place API.
