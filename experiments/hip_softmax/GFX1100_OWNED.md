# gfx1100 private attention-score reuse

This experiment extends the qualified OpenFold3 N1000 wave32 softmax, conditioning
reuse and triangle-copy composition. It is not a maintained production profile.

## Change and ownership contract

The pinned `AttentionDispatcher._fused` produces a fresh QK `einsum` result with
no other consumers. The candidate overwrites that tensor with probabilities,
removing the separate allocation and deterministic initialization of its output.
QK, PV, mask/bias additions, FP32 math, reduction order and wave selection stay
unchanged. Each wave loads its entire row before writing it; rows are disjoint.

`gfx1100_owned.hip` has a separate C entry requiring identical input/output
pointers. The old out-of-place library remains immutable. The wrapper also
requires inference mode, strict determinism, memory filling, contiguous full
storage, N1000, and storage disjoint from both mask and bias. Storage checks alone
cannot establish private ownership: this wrapper is only valid at the audited
fresh-QK call site. Ineligible storage uses the established out-of-place path.

## Reproduction and gates

Precision: `/media/990Pro/fireprot`, HIP0 / W7900 / gfx1100 / PCI `0000:53:00.0`.
Build with `build_gfx1100_owned.py` through Rust, with GPUs hidden and
`-fno-fast-math -ffp-contract=off`. The library, worker tree, requests, profiles,
Rust CLI and bridge must be pinned before measurement.

1. `screen_gfx1100_owned.py ROOT`: batches 1/8/232/256, both bias layouts,
   zero/partial/nonzero masks, NaN/Inf/signed-zero inputs, non-default stream,
   offset/overlapping storage fallbacks. Finite results must match native Torch
   bitwise; all cases must match the old HIP path bitwise.
2. `prepare_gfx1100_owned.py REPO ROOT`: freeze both arms from
   `gfx11-of3-copy-20261019-v1`; vary only the owned-score switch.
3. Rust `measure-inference ROOT/measure-cycle.json ROOT/measurement-cycle`,
   followed by `audit_gfx1100_owned.py REPO ROOT ROOT/audit-cycle.json`.
4. Only after that audit passes, Rust `benchmark-pairs ROOT/pairs.json
   ROOT/paired`, followed by the independent final audit and collector.

Both arms use strict FP32, N1000, R3/four trunk passes, D200/S1, seed101,
query-only MSA and no supplied templates. The original maintained native
prediction remains the reference for all 11 arrays and complete CIF bytes.
Each candidate must submit 1,696 owned softmax calls with zero storage fallbacks;
the existing 3,440 HIP triangle copies and conditioning reuse remain active.

Complete-attention microbenchmarks include native QK/PV and allocation/filling.
They are not model E2E measurements. Model allocator peaks cover forward only.
Fresh-worker paired tests retain unflushed caches and unlocked clocks; inspect
device occupancy and telemetry, and never multiply earlier incremental ratios.

## Qualified N1000 result

Three pairs: 231.100820 -> 223.261027 s E2E, 1.035115x incremental.
All seven predictions pass the original-native arrays/CIF gate. Full-model
allocator peaks are unchanged. See [the evidence](../../provenance/precision_openfold3_owned_gfx1100_20261019.json)
and [host status](../../docs/precision-setup.md#exact-gfx1100-private-qk-score-reuse-2026-10-19).
