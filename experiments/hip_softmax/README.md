# Experimental FP32 HIP attention softmax

This is a new OpenFold3 experiment, not a promoted production backend.
The user requests handwritten HIP / CK-style optimization; Triton enablement
is outside this workstream. Rust owns native builds, publication, environment
inventories, model supervision and numerical acceptance. Python files here are
the explicit Torch tensor boundary and operator diagnostics.

## Arithmetic and scope

`fused_softmax.cpp` combines native-order mask and pair-bias additions with
softmax for FP32 `[B,4,N,N]` scores at N1000/N1536. It follows the installed
Torch `PersistentSoftmax.cuh`: 16/32 lane-local slots, wave64 XOR reductions in
32/16/8/4/2/1 order, `std::exp`, division and the zero-sum NaN branch.
No fast-math compiler option is requested. QK and PV remain upstream einsums.
All score/base address arithmetic uses 64-bit offsets. The boundary checks
dtype, shape, device, contiguity, bias broadcasting and inference mode.

The existing BF16 AtlasFold kernel supplies a row-scheduling reference; its BF16
rounding semantics are not copied into FP32. The user-provided local
`docs/isa/CDNA2.pdf` was reviewed with a GPT-6-luna xhigh agent. Relevant PDF pages
are 91–92 (addressing), 133 (`V_EXP_F32` is base-2 and flushes denormals), and 180
(`DS_BPERMUTE_B32` and EXEC). These correspond to printed pages 83–84, 125 and 172.
The ISA does not establish performance or equivalence of `V_EXP_F32` to `std::exp`.

## Completed operator checks

- `screen.py`: 36 bitwise-equal configurations, including zero/partial/nonzero
  masks, special floating values, both sequence lengths and a >4 GB N1536 score.
- `time_operator.py`: every variant is compared bitwise before timing, for
  starting and ending bias layouts. Two warmups and five GPU-event observations
  follow, alternating variant order. QK/PV and native score reset are excluded;
  output allocation and candidate bias packing are included.
- Code object metadata: wave64, no fixed LDS or scratch allocation; N1000 uses
  36 VGPRs/56 SGPRs and N1536 uses 57 VGPRs/48 SGPRs. These are compiler counts,
  not an occupancy measurement. Disassembly contains DS_BPERMUTE and 64-bit
  address carry operations.

Latest isolated measurements on HIP5 / PCI b3 / SMI7, gfx90a:

| Shape / bias layout | Native median | Packed HIP median, 2 waves | Ratio |
| --- | ---: | ---: | ---: |
| N1000 B256, starting | 33.811 ms | 10.973 ms | 3.08x |
| N1000 B256, ending | 52.841 ms | 11.006 ms | 4.80x |
| N1536 B128, starting | 39.974 ms | 9.885 ms | 4.04x |
| N1536 B128, ending | 85.644 ms | 10.269 ms | 8.34x |

These synthetic-input operator results are not model or E2E gains. The selected
candidate packs the O(N^2) bias and reorders independent rows for reuse across
eight outer rows, without changing arithmetic within each row.

## Full-model experiment and retained setup failure

The numerical/benchmark source deployment is `/tmp/fireprot-hip-softmax-v1` on
DiamondHill. The successful Rust build cache ends in
`44fe1e728065b85aab6ed52ce947694423a167f1b8e663a529569790bd5801da`.
The current immutable bundle is
`/media/IntelSSD/fireprot/runtime/kernels/experimental-softmax-f32-v2`, manifest
SHA-256 `c67bca20fd630e0e6f3e688a2ae24b71df0a5571be057c4896d19bb1d541b47d`.

An ISA extraction command omitted an output ELF path, causing llvm-objcopy to
rewrite the first published library's host ELF. A later loader rejected its hash;
the first N1000 model failed final environment revalidation, and the first N1536
model was explicitly cancelled. Both are excluded from accepted correctness and
timing results. The build cache remained untouched. A fresh v2 bundle restores
the exact original library. Subsequent extraction uses a distinct output copy;
the original/extracted device fatbins match, and the v2 library hash is unchanged.
Do not use or repair the invalid v1 bundle in place.

The current model deployment is `/tmp/fireprot-hip-softmax-model-v2`. It copies
the maintained worker/shim/kernel trees, applies `worker-overlay.patch` with
`git apply -p0`, adds `boundary.py` and `worker_overlay.py` alongside the workers,
and places the HIP source in its kernel identity tree. Both arms load the same
Rust-verified bundle. A new strict environment inventory binds the bridge and
published bundle in addition to the existing OpenFold3 environment.

N1000/N1536 full-work candidate validation passes all 11 arrays bitwise against
strict native references: FP32, R3, D200, S1, seed101, original source and weights.
Successful HIP selections are 1,702 and 5,592 respectively. Calls count function
entries; selections are counted after the HIP boundary returns, so tuning errors
before that point can prevent the dispatch counters from summing to all entries.
Evidence is in `provenance/openfold3_hip_softmax_firstpass.json`.
The candidate flag is
`FIREPROT_FUSED_SOFTMAX`; it is absent from maintained production profiles.
Three sequential N1536 AB/BA/AB pairs are running on HIP7/PCI93/SMI5 with no
other FireProt GPU job launched during the experiment. Those gates and E2E
measurements remain required before promotion. Initial candidate validations
ran concurrently on separate GCDs and do not establish E2E gains.
Never modify worker trees, inventories or bundles during the paired runs.

## Partial paired evidence

The first N1536 pair passes all 11 arrays bitwise and CIF atom identity checks.
Complete invocation time is 953.331 -> 694.973 seconds (1.37175x); model time is
858.251 -> 599.567 seconds (1.43145x). Rust reports `insufficient_repeats`.
The other two pairs are still running. The immutable first-pair report is pinned
in `provenance/openfold3_hip_softmax_pair_000.json`; no aggregate verdict or
production promotion follows from this one pair.

## Maintained integration requirements

Read-only inspection confirms that the installed upstream
`openfold3/core/model/primitives/attention.py` (SHA-256
`30fe5433982ee73e3d7ce5341fa6b510b1ea332dbde072d56129632cc53f0746`)
calls the module-level `_attention` function from multiple attention classes.
The experimental global function replacement must not become an unrestricted
public hook. A maintained integration must scope dispatch to the active model
invocation, preserve unrelated callers, restore hooks on failure, and reject
conflicting/nested installations it cannot safely support.

Rust request validation currently admits native/Triton OpenFold3 requests, while
the worker only executes native OpenFold3. Enabling a HIP route requires explicit
changes at both boundaries. The existing `HipFused` enum value can be reused, but
schema admission alone is not integration. Rust's `OperationRegistry` must own
selection and counts, with tensor eligibility recomputed before every call and
included in cache keys. Pin the observed Torch/HIP and gfx90a compatibility,
upstream arithmetic source, and published library manifest. Unsupported tensors
must use the unchanged native path; kernel failures after launch must surface.

Before rollout, validate the maintained route's own N1000/N1536 outputs and
paired performance: the experimental worker results do not automatically cover
different dispatch overhead, artifact names, hook lifetimes or fallback behavior.
In-place storage reuse and diffusion caching remain separate experiments.

Local integration preparation now includes
`python/fireprot/backends/amd/softmax_fp32.py` and an exact source copy at
`kernels/amd/fireprot_softmax_fp32.cpp`. Twelve CPU/meta tests cover tensor
contracts, real bias strides, broadcast masks and rejected inputs. The loader
requires a new pinned `fireprot_softmax_fp32` / `softmax-fp32-v1` publication,
the measured Torch/HIP version and gfx90a; no such maintained publication has
been validated yet. No model adapter imports this boundary, no new request
backend is enabled, and source-copy identity is not a compiled-artifact or
maintained-model acceptance result.
