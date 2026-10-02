# Protenix AMD performance report

Status: measured development result, 2026-10-02.

## Scope

This report covers Protenix Mini-default v0.5.0 on one AMD Instinct MI250X/MI250 gfx90a GCD. The input is `cases/1izr_L124/input.json`, a 124-token single-chain development case. Results apply to FP32 inference with four cycles, seed 101, one sample, warm in-process caches, no MSA/template search, TF32 disabled, and one or five diffusion steps.

Source identities:

- FireProt: `d31013ac643bdb7e033dd378795d216e29ecf448`
- independent Protenix integration commit: `316f5778e80965015a80339de632291f1abbb7f5`
- Protenix upstream baseline: `d3b4db6a121dd4584edd85e93744239325a2b72e`
- profiler SHA-256: `812972eb94817a421172630d3bf839eece467361d48bf2215fbec9771b229eb3`

The runtime used PyTorch `2.12.0a0+git78258b9` with HIP `7.14.60850`. Every reported performance run passed the exclusive-device preflight at 0% GPU use and approximately 11 MB VRAM use before model load. Each workload used five warmups and thirty measured repetitions in one process. GPU event timing excludes preprocessing and serialization.

## Promoted implementations

| Operation | Implementation | Exact promoted capability |
| --- | --- | --- |
| Triangle multiplication | `amd-short-sequence-torch` | gfx90a, FP32, inference/no-grad, contiguous unbatched pair tensor, `N <= 256`, in-place-safe Protenix path |
| Triangle attention | `amd-gfx90a-efficient-sdpa` | gfx90a, FP32, inference/no-grad, QKV `[124,4,124,32]`, observed strided layout, two observed Protenix additive bias shapes |

Triangle attention uses ROCm memory-efficient SDPA and an outer batch chunk of 64. The chunk bounds the materialized combined-bias workspace and retains a measured latency benefit. Unmatched signatures fall back to native Protenix.

## End-to-end performance

| Workload | Metric | Native | FireProt | Native / FireProt |
| --- | --- | ---: | ---: | ---: |
| C4/S1 | model median | 453.53 ms | 373.94 ms | 1.213x |
| C4/S1 | model p95 | 457.51 ms | 406.06 ms | 1.127x |
| C4/S1 | Pairformer median | 397.19 ms | 320.71 ms | 1.238x |
| C4/S1 | triangle multiplication median | 193.27 ms | 135.78 ms | 1.423x |
| C4/S1 | triangle attention median | 135.19 ms | 111.73 ms | 1.210x |
| C4/S5 | model median | 518.43 ms | 437.86 ms | 1.184x |
| C4/S5 | model p95 | 541.45 ms | 468.27 ms | 1.156x |
| C4/S5 | Pairformer median | 393.31 ms | 316.69 ms | 1.242x |
| C4/S5 | triangle multiplication median | 190.07 ms | 134.16 ms | 1.417x |
| C4/S5 | triangle attention median | 134.55 ms | 111.20 ms | 1.210x |

Each FireProt workload selected each promoted implementation 5,040 times. The native runs selected no FireProt operation.

## Peak memory

| PyTorch peak metric | Native maximum | FireProt maximum | Difference |
| --- | ---: | ---: | ---: |
| Allocated | 850,018,304 B | 837,647,360 B | -12,370,944 B (-1.46%) |
| Reserved | 1,029,701,632 B | 1,000,341,504 B | -29,360,128 B (-2.85%) |

The first unchunked SDPA candidate was faster but increased peak allocated memory by approximately 15.8 MB and was rejected. Chunking the outer batch at 64 removed that regression before promotion.

## Numerical validation

Native and FireProt Mini-default outputs used the same input, seed, FP32 dtype, four cycles, sample count, preprocessing, and cache policy.

| Workload | Atoms | Aligned coordinate RMSD | Maximum coordinate delta | Maximum confidence delta | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| C4/S1 | 945 | 1.0758e-4 A | 9.7656e-4 A | 3.4332e-5 | pass |
| C4/S5 | 945 | 6.0290e-5 A | 2.9278e-4 A | 3.8147e-5 | pass |

The locked coordinate and confidence thresholds are `1e-3`. Experimental-coordinate utilities use only `coordinates[mask]` from `gt_mapping.npz`; `reference` is not treated as experimental ground truth.

## Residual profile

A final `rocprofv3` profile of the C4/S1 FireProt path recorded 14,809 kernel dispatches and 349.317 ms aggregate kernel duration. The category shares were:

| Category | Share | Calls |
| --- | ---: | ---: |
| GEMM | 37.94% | 3,143 |
| Elementwise | 20.09% | 4,807 |
| Copy/layout | 16.83% | 4,852 |
| LayerNorm | 15.05% | 805 |
| Other, including efficient attention | 6.28% | 709 |
| Reduction | 3.64% | 416 |
| Softmax | 0.17% | 77 |

The trace also recorded 1,851 host-to-device copies taking 52.895 ms, 178 allocation calls, and 87 free calls. The dominant residual targets are therefore GEMM selection, LayerNorm, elementwise epilogues, and layout copies. No additional implementation was promoted because these categories require isolated numerical and end-to-end gates rather than a global switch.

## Limits

- The eight repository cases are development inputs, not an independent test set. The repeated performance result above uses one 124-token case.
- Base weights, longer sequences, complexes, ligands, production diffusion-step counts, BF16/FP16, gfx942, and multi-GPU execution are unmeasured.
- BLAS microbenchmarks found different winners by signature; a single global hipBLASLt, hipBLAS, or CK choice is not generally optimal.
- The installed environment does not provide the optional `triton` package required by Protenix's existing `triattention` implementation. FireProt uses PyTorch ROCm efficient attention for the promoted signature.
- Raw manifests, predictions, and profiler traces remain on DiamondHill under excluded `outputs/` paths. The compact machine-readable record is in `provenance/protenix_amd_benchmark.json`.
