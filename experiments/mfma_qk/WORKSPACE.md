# Private score workspace experiment

This experiment retains one initialized FP32 score tensor per `WorkspaceFusion`
instance. It uses the accepted prefetch-v11 kernel, original Q/K packing and
native PV. Strict deterministic execution and uninitialized-memory filling stay
enabled. The first `torch.empty` receives its normal NaN fill; subsequent calls
reuse allocated storage and fully overwrite its scores. This is an explicit
change in workspace/cache lifetime, not an equivalent allocation policy.

## Ownership and bounds

- Only independently screened B32/B256, N1536, four heads and K32 are accepted.
- The caller supplies a byte budget, capped at9GiB; the shape stays fixed.
- A workspace binds to its first device and stream and its constructor thread.
- Concurrent use, other threads/streams, graph capture, training and relaxed
  determinism/filling settings are rejected.
- Q/K packing, optional bias-contiguous conversion and native PV are inside the
  call. Only PV output escapes; a storage check rejects alias with the scores.
- Same-stream launch ordering places PV reads before any subsequent overwrite.
- Explicit close synchronizes the owning device and releases the retained tensor.
- A submission-path error prevents further calls through that instance.

These are standalone tensor-boundary experiments. There is no shared framework
workspace manager or model adapter integration. Rust should own any future
resource policy; model lifecycle and tuner compatibility must be established
before selecting this in inference.

## Validation and measurement

`benchmark_workspace.py CONTROLS NEW_DESTINATION --batch 32` first tests three
seeds101/102/103 and both bias layouts, including a fully masked outer item.
Before reuse it poisons the old scores with NaNs or a finite sentinel. Every
probability bit and PV output bit is compared with native Torch. Retained prior
public results must remain unchanged. GPU guards reject a second thread and
stream without submissions/reallocations. First-allocation call time and Torch
allocation peak are recorded separately from the warm measurements.

Five arms are interleaved in balanced cyclic/reversed order: native, split HIP,
fresh fusion, retained workspace, retained workspace with bias normalization.
All packing and any bias conversion remain inside timing. Every warm and timed
output is checked bitwise outside the interval. There are100 timed calls per
batch. Each workspace must report one allocation and30 calls, then release its
entire retained tensor. Two workspaces persist during all arms, adding2.25GiB
at B32 or18GiB at B256. This common retained memory is included in reported live
and peak Torch allocations; it does not establish a model memory benefit.
No cache flush or continuous device isolation monitor is provided by this
standalone script. A controller records idle-device preflight.

CPU tests cover byte-budget rejection before library loading, thread ownership,
concurrency/closed-instance rejection, strict-setting enforcement and rejection
of non-HIP operands before dispatch. GPU correctness and timing results belong
in separate pinned provenance; CPU tests do not establish GPU acceptance.

## Completed B32/B256 result

Both batches pass twelve full-score/PV qualifications, ten prior-result
immutability checks, thread/stream rejection gates,100 timed-output checks,
and exact allocation/release accounting. Each instance allocates once, serves
30 calls and releases its entire retained tensor. Six CPU rejection tests pass.

| Batch / bias | Split HIP (ms) | Fresh fusion (ms) | Workspace (ms) | Workspace + contiguous bias (ms) |
| --- | ---: | ---: | ---: | ---: |
| 32 / contiguous | 5.528 | 6.196 | 5.359 | 5.360 |
| 32 / transposed | 5.827 | 8.328 | 7.482 | 5.663 |
| 256 / contiguous | 43.469 | 48.942 | 41.914 | 42.093 |
| 256 / transposed | 43.774 | 65.783 | 58.702 | 42.384 |

The normalized workspace reduces warm B256 time3.17% contiguous and3.18%
transposed versus split HIP. First workspace calls with contiguous bias cost
49.13/49.19ms at B256 (6.55/6.31ms at B32), including initial allocation/fill;
these are single observations with other Torch operations already warm.
The retained byte cost remains9GiB per B256 instance. Model peak memory,
chunk tuning, lifecycle and E2E benefit are untested, so no model path is changed.

Evidence: `provenance/openfold3_qk_workspace_benchmark.json`. Remote archive
`qk-workspace-v1-results.tar.gz`, SHA256
`1ee16f1fb30c8afb315d2920b988dc1ea3b9e32b6299b379c7f4b22034100d54`.
