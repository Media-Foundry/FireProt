# Single-item OpenFold3 data preparation experiment

This experimental context runs one CPU batch in the main process while invoking
the original Lightning worker-0 initializer with the DataLoader iterator's actual
base seed. It restores parent Python, NumPy, Torch CPU/CUDA RNG, thread count and
deterministic settings before model execution. CPU preparation uses the one-thread,
non-deterministic-algorithm context of a fresh data worker; model execution retains
strict deterministic FP32 and memory filling.

## Scope and usage

`isolated_scope.py` exports `isolated_data_scope(output)`. In the frozen worker,
copy it to `python/fireprot_workers/isolated_data_scope.py`, set the runner's
`data_module_args.num_workers` to zero and wrap the upstream CLI call in the
context. The context requires a single-item dataset and a single batch, an explicit
DataLoader generator and worker initializer, and CPU output tensors. It restores
its temporary DataLoader hook in `finally`. It is not safe for concurrent loader
execution in the same process and is not a maintained-profile default.

This is a version-pinned experiment using private Torch iterator APIs. Do not
extend it to multiple inputs, seeds or distributed ranks without separate
qualification. Plain `num_workers=0` failed model parity; bypassing worker RNG
initialization is not equivalent to this context.

## Qualification

OpenBind-0, strict FP32 N1536, R3/four trunk passes, D200/S1, seed101, HIP7,
qualified gate/packing composition, default unchanged model execution:

- All eleven output arrays are bitwise equal to the qualified anchor.
- Complete CIF bytes are equal.
- The single data-preparation call records base seed `302501347963651595`,
  worker ID 0, one CPU thread and restoration of parent strict mode/128 threads.
- One observation is 351.541 s complete invocation / 256.046 s model forward;
  it is not used as a speedup comparison. Three matched pairs are now complete.

`test_scope.py` is a CPU regression for parent-state restoration on both success
and a dataset exception. It also covers Lightning-style batch samplers whose
`batch_size` attribute is `None`. It passes in the pinned model Python environment:

```bash
HIP_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /path/to/openfold3-python test_scope.py
```

See [qualification and archive](../../provenance/openfold3_isolated_data_n1536.json).
Frozen paired entry: DiamondHill
`outputs/experiments/openfold3-isolated-data-pairs-20261009/pairs.json`, using the
CLI pinned in `preparation.json`. Only `FIREPROT_ISOLATED_DATA=0/1` and the
associated profile hash differ. Both profiles use the same newly audited
2026-10-09 environment inventory. Use fresh output identifiers for reruns.


## Three matched E2E pairs

Strict N1536 FP32 R3/D200/S1, seed101, HIP7, fresh processes and unflushed caches;
shared gate/packing source and the same audited 2026-10-09 environment in both
arms. The only declared treatment is `FIREPROT_ISOLATED_DATA=0/1` plus its profile
hash. Complete invocation medians are **349.971 → 341.782 s (1.0240x)**,
**2.34% less time**. All three pairs improve; all six outputs pass eleven bitwise
array checks and complete CIF byte comparisons. Model-forward medians are
254.622 → 254.178 s, so the E2E improvement is primarily outside forward.

Retain this route in the frozen single-item experimental composition. Maintained
profiles and broader input support remain unchanged. This is an incremental
comparison, not a new direct original-Torch result.
See [paired evidence and durable archive](../../provenance/openfold3_isolated_data_e2e_n1536.json).
