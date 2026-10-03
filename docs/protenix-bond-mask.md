# Direct int64 bond-mask construction

## Implementation and contract

Independent Protenix patch 0008 changes `Featurizer.get_mask_features` to
allocate its private NumPy bond matrix with `dtype=np.int64`, populate the
same symmetric binary entries, and return `torch.from_numpy` storage.
The previous implementation allocated float64, copied to a Torch float32
matrix, then copied to int64. Neither intermediate floating matrix contributes
information: every written element is exactly zero or one.

The final `[N_atom,N_atom]` contiguous CPU int64 tensor, field presence and
all other mask fields are retained. Each invocation owns a new backing array;
returned tensors from separate calls do not alias. The tensor retains its
NumPy owner after function return. No bond classification, atom ordering,
random draw or model/recycle arithmetic changes. This is a CPU data-path
optimization; it does not reduce the final tensor's GPU memory footprint.

The method has one direct caller, `get_all_input_features`, which is shared
by inference JSON featurization and training feature generation. That shared
scope is why this patch preserves the full mask rather than removing its
field. Existing training-loss consumers continue receiving the same values.
Training throughput and training model behavior have not been benchmarked.

## CPU pipeline measurements

Source base `70a6d41`, pre-patch Protenix `932de05`, integrated Protenix
`bb3be99`. Mini-default v0.5.0 configuration, native feature dtypes, FP32
configuration, seed 101 reset before each call, one CPU thread, no
MSA/RNA MSA/template/ESM search. The previously validated unused-template
option is enabled on both sides. One unmeasured warmup per length, then three
pairs in alternating native/candidate order; process and filesystem caches
are warm. No GPU initialization, checkpoint, cycles, steps or prediction
samples apply to this CPU feature-only experiment.

| Length | Native feature median | Direct int64 median | CPU feature speedup |
| --- | ---: | ---: | ---: |
| 1000 | 1.337 s | 0.901 s | 1.484x |
| 1536 | 2.634 s | 1.673 s | 1.574x |

A second run against the integrated source confirms medians of
1.343 → 0.911 s and 2.656 → 1.692 s. Hashing occurs outside the timed region.
Every feature tensor and metadata field (including `bond_mask`), atom
coordinates/annotations/bonds, and Python/NumPy/Torch CPU RNG state matches
exactly across all pairs. There are no excluded feature fields in this test.
The final bond-mask allocation remains 509,315,528 bytes at N1000 and
1,178,842,568 bytes at N1536. Lazy zero-page allocation contributes to the CPU
benefit; subsequent host-to-device transfer may still have to fault/read
those pages. Full-request gains must therefore be measured separately.

Synthetic tests cover empty bonds, duplicate/reversed edges, a self edge,
and boundary indices. They verify exact values, dtype, layout, unchanged
other mask fields and independent call ownership. The local suite passes
157 tests with 53 HIP-gated skips. Production-method tests load its AST to
avoid optional local data dependencies; the remote pipeline runs import the
actual complete upstream modules.

The benchmark reconstructs the native expression when run on the integrated
source, verifies exact expected source patterns, and swaps only this method
inside its single-process CPU experiment. It restores the installed method
on exit. No temporary patching is installed in model runtime.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_bond_mask.py \
  --output outputs/benchmarks/bond-mask.json
```

[Evidence](../provenance/protenix_bond_mask_cpu_70a6d41.json) contains both
runs, source/input hashes, exact candidate source, per-call stage timing,
full-feature/atom/RNG digests, tensor size and GPU-initialization checks.
No E2E speedup, model-output comparison, or peak CPU RSS reduction has yet
been measured for this increment. Complete inference-only mask omission
remains a separate candidate requiring consumer and model validation.
