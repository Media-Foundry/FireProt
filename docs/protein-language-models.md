# Standalone protein language models

For model-resident multi-sequence inference, explicit GPU microbatching and its
separate numerical/performance contract, see [pLM batches](plm-batch.md).
The single-sequence results below retain their original scope.
For the latest ESMC/ESM2/ESM3 exact operator and loading route, see
[ESM optimization](esm-optimization.md).

FireProt provides separate Rust commands for protein language model inference:

```text
fireprot esm REQUEST.json PROFILE.json
fireprot atlaslm REQUEST.json PROFILE.json
```

The commands validate the model family before starting the same pinned, monitored
worker lifecycle used by `infer`. `esm` selects ESMC, ESM2 or ESM3;
`atlaslm` selects AtlasLM. These commands perform **sequence forward inference**;
they do not invoke folding, diffusion, structure generation or iterative sequence
sampling. `infer` also accepts the same `plm` requests.

## Contract

The initial interface accepts one monomer FASTA or the existing single-chain
Protenix JSON input, 1–1,000 residues, using the 20 standard amino acids or `X`.
It does not truncate inputs or silently split long sequences. The optimized
route is qualified only for the N1000 workload on gfx90a with the recorded
ROCm/PyTorch build. Native inference supports the declared shorter inputs.
Multiple sequences use the separate explicit [batch contract](plm-batch.md).
Masked-token/generation tasks are not connected.

```json
"workload": {
  "model": "plm",
  "model_name": "esmc-fold2-6b",
  "precision": "native"
}
```

| Request model name | Native outputs | Precision | Weights |
| --- | --- | --- | --- |
| `esmc-fold2-6b` | Tokens and final normalized embedding | BF16 embedding | Encoder bundled in the pinned Biohub ESMFold2 release; no MLM head |
| `esm2-3b` | Tokens, final normalized layer-36 embedding and logits | FP32 | `esm2_t36_3B_UR50D.pt` plus contact regression companion |
| `atlaslm-3b` | Tokens, native embedding and logits | FP32 embedding, BF16 logits | `atlaslm_3b_base.pth` |
| `esm3-open` | Tokens, native prenorm embedding and sequence logits | BF16 embedding/logits with native BF16 autocast | Official `biohub/esm3-sm-open-v1`; N1000 native factory parity passed; prepared profile uses the faster official factory |

Tokens retain BOS/EOS: `[1, N+2]`. Embeddings retain those positions too:
`[1, N+2, hidden_size]`. For residue-only use, select `[:, 1:-1, :]`.
Logits retain the complete native vocabulary; they are not probabilities or
perplexity scores. Hidden sizes are 2560 / 2560 / 2304 / 1536, respectively.

Schema-5 output bundles bind the retained input bytes to the request SHA-256 and
validate every token ID, shape, dtype and finite floating value in Rust. BF16
NPZ arrays use lossless `uint16` storage with their original dtype declared in
`predictions.json`. They require reinterpretation as BF16, not numeric conversion
of the stored integers. Language model outputs have no CIF or folding confidence.
Existing folding output schemas and their accuracy gates remain separate.

## Prepared DiamondHill examples

Use a release CLI built from this checkout. The profiles refer to an experimental
DiamondHill deployment, not a portable installer. Copy each request and choose
an unused `run_id` and `output_dir` before executing it.

```bash
fireprot esm configs/requests/plm-esmc-fold2-6b-n1000-fireprot.json \
  configs/diamondhill/plm-esmc-fold2-6b-optimized.json
fireprot esm configs/requests/plm-esm2-3b-n1000-fireprot.json \
  configs/diamondhill/plm-esm2-3b.json
fireprot atlaslm configs/requests/plm-atlaslm-3b-n1000-composed.json \
  configs/diamondhill/plm-atlaslm-3b-optimized.json
fireprot esm configs/requests/plm-esm3-open-n1000-native.json \
  configs/diamondhill/plm-esm3-open.json
```

The request's `hip_device` controls GPU selection. The supervisor replaces inherited
visibility; setting `HIP_VISIBLE_DEVICES` alone does not override that field.
HIP6 maps to PCI `0000:8e:00.0` / SMI4, HIP7 to `0000:93:00.0` / SMI5 on the
qualified boot. Device identity is probed again for every invocation.

ESM2 uses an isolated fair-esm environment because fair-esm and Biohub ESM both
export the Python package `esm`. Its request pins all Python files in the
installed fair-esm package. Other adapters bind imports to independent Git sources.
No worker downloads a missing checkpoint. Weights stay outside the repository.

## Optimization scope

- **ESMC:** one HIP kernel computes Q/K rotary embeddings, replacing separate
  multiplication, sign, concatenation, addition and copying kernels. Both BF16
  products are rounded before the final addition, preserving native arithmetic.
- **AtlasLM:** the same arithmetic uses its 36-head layout and native RoPE tables;
  invariant table casts are prepared once for the shared rotary module. The
  composed request also uses HIP SwiGLU with the native intermediate BF16 rounding
  and eager GPU checkpoint restoration with strict assignment. Published tensors
  must already be BF16; native RoPE buffers are recreated and remaining meta
  tensors are rejected. No language model result or stochastic folding pass is cached.
- **ESM2:** omit attention-head means discarded by this sequence-only caller.
  Checkpoint-covered meta construction avoids random parameter initialization.
  Published FP16 weights are still promoted to FP32, as in the native loader;
  strict state loading and absence of meta tensors are checked before execution.
- **HIP Graph screen:** exact ESMC outputs, but only 1.024x warm speedup and about
  0.50 seconds capture overhead. It is not enabled in the standalone CLI.

Each opt-in scope uses the Rust operator registry and checks completed module
coverage. HIP builds use Rust's `build-hip` service; requests pin the resulting
library. Unsupported tensor signatures retain native execution, while incomplete
coverage prevents a run from being accepted as the qualified optimized route.

## Preliminary warm-forward measurements

These are diagnostic same-process N1000 measurements, with explicit warmup and
AB/BA/AB ordering, not complete process invocation E2E speedups. ESMC uses HIP6;
AtlasLM uses HIP6; ESM2 uses HIP7. The 80-layer ESMC and 48-layer AtlasLM trained
RoPE calls were checked individually. Every retained output array is bitwise
identical across the compared model paths.

| Model | Native → candidate median | Speedup | Candidate |
| --- | ---: | ---: | --- |
| ESMC bundled 6B | 196.981 → 173.993 ms | 1.132x | HIP RoPE, 30 observations per arm |
| AtlasLM-3B | 104.976 → 95.312 ms | 1.101x | HIP RoPE, 15 observations per arm |
| ESM2-3B | 243.651 → 238.161 ms | 1.023x | Omit unused attention mean, 15 observations per arm |

The [screen evidence](../provenance/plm_warm_screens_20261015.json) records all observations and the Rust HIP build pin. The screen adapters precede maintained Rust-registry dispatch packaging. They do
not establish a cold-start speedup. Complete invocation results below include startup, asset verification and output acceptance. No folding E2E gain is claimed from these standalone results.

## Complete invocation E2E results

### Latest ESMC composition

Three additional AB/BA/AB N1000 pairs measure **36.047 → 23.951 seconds
(1.505x, 33.56% less time)** for serial full-asset verification/native Torch
versus four-thread full-asset verification/HIP RoPE. Every archive and both
output arrays pass the unchanged acceptance gates and bitwise comparisons.
HIP6, native BF16, seed101, one sequence forward, fresh processes and unflushed
filesystem/GPU library caches are identical across arms.
[Full observations and receipt pins](../provenance/plm_esmc_parallel_hash_e2e_20261015.json).

The gain mainly comes from **CPU startup verification of the six large shards**.
Each declared file is still read completely and checked against SHA-256 on every
invocation. No digest cache, model-output cache, reduced weight set or skipped
integrity check is used. All verification threads finish before worker launch.
First-forward medians are 1.038 → 1.011 seconds; the 1.505x result must not be
described as a 1.505x transformer/GPU speedup.

Use `configs/diamondhill/plm-esmc-fold2-6b-optimized.json` with a fresh copy of
the FireProt request. Its profile sets `FIREPROT_PREFLIGHT_HASH_THREADS=4` and
uses this matched CLI/worker/bridge deployment:

```text
/media/IntelSSD/fireprot/outputs/experiments/plm-e2e-next-20261015/target/release/fireprot
```

The explicit profile setting accepts 1–4 threads; omission preserves serial
verification. It applies to independent input/checkpoint/resource files, not
parallel chunks of a single SHA-256 stream. It is not a reason to expect the same
gain for single-checkpoint AtlasLM or ESM3. The launch environment and profile
hash bind the setting to the comparison. Reproduce with
`experiments/plm/prepare_hash_pairs.py` in a new built source snapshot.

### Earlier operator-only comparison

Three fresh-process AB/BA/AB pairs per model completed on HIP6, N1000, using a
single frozen Rust CLI and serial execution. Native precision, one forward,
seed101 (ESMC/ESM2) or seed102 (AtlasLM), unflushed filesystem/GPU library caches;
no observations were discarded. All eighteen output archives passed Rust
acceptance. All native tokens, embeddings and available logits are bitwise equal.
The read-only collector rehashed every accepted metadata/output file and compared
all retained NPY bytes. [Versioned evidence](../provenance/plm_e2e_n1000_20261015.json).

| Model | Native → opt-in E2E median | Ratio | Decision |
| --- | ---: | ---: | --- |
| ESMC bundled 6B | 36.148 → 36.798 s | 0.982x | RoPE alone gives no cold-start E2E gain; superseded by the composition above |
| ESM2-3B | 39.304 → 18.827 s | **2.088x** | Opt-in meta initialization plus omitted unused attention mean; exact FP32 outputs |
| AtlasLM-3B | 15.913 → 16.219 s | 0.981x | RoPE alone gives no cold-start E2E gain; superseded by the composition below |

The RoPE optimization remains available explicitly for experimentation, with its
warm-forward benefit scoped separately. Loading, hashing and process startup
outweigh the tens of milliseconds saved in one ESMC/AtlasLM forward. ESM2's
substantial E2E gain comes primarily from avoiding random initialization of
parameters that the published checkpoint immediately overwrites. This is not a
2.088x transformer-kernel speedup.

The frozen benchmark CLI is:

```text
/media/IntelSSD/fireprot/outputs/experiments/plm-20261015/qualified/target/release/fireprot
```

The profiles for these three models use the matching frozen worker directory and
bridge, except the separate ESMC and AtlasLM `-optimized` profiles. Inference examples may point to a previously used output directory;
always copy requests and change their run ID/output path before execution.

### AtlasLM loading and HIP composition (2026-10-16)

The final three AB/BA/AB pairs improve complete invocation **15.915 → 15.046 s
(1.058x, 5.46% less time)**. These fresh-process N1000 comparisons use HIP6,
seed102, native BF16 weights/autocast, FP32 embeddings and BF16 logits, one
forward, unflushed filesystem/GPU library caches, and identical assets. Every
archive passes Rust acceptance and all three output arrays match bitwise.
Each candidate records 48 HIP RoPE and 48 HIP SwiGLU calls, with one shared
rotary module. [Final E2E evidence](../provenance/atlaslm_composed_e2e_20261016.json).

The native profile attributes 57.15% of device time to GEMMs and 12.26% to
attention. Those computations and the native attention mask remain unchanged.
Both attempted unmasked attention routes changed outputs and were rejected.
The accepted changes are:

- Restore checkpoint tensors directly to the GPU with eager deserialization,
  then strictly assign them to the meta-constructed model. This avoids CPU mmap
  faults during hundreds of individual native copies and avoids a second GPU
  allocation for every parameter. No checkpoint or input hash is skipped.
- Prepare the shared HIP RoPE module once instead of reinstalling it 48 times.
- Fuse SiLU and multiplication in a handwritten HIP kernel, preserving BF16
  rounding between the operations. All 48 trained calls match, and an additional
  test of all 65,280 finite BF16 bit patterns with varying gates has zero raw-bit
  differences, including signed zero and overflow results.

The first eager-loading composition copied the GPU state into separate model
storage. It measured 16.100 → 15.053 s (1.070x) in its own three-pair comparison,
but doubled loading peak allocation from 6.199 to 12.361 GB. The final strict
assignment route returns the measured loading peak to **6.194 GB**. Its candidate
E2E median is similar; separate native medians vary between experiments. The
published ratio uses the final matched comparison, not the larger earlier ratio.
[First composition](../provenance/atlaslm_eager_copy_e2e_20261016.json).

A separate same-process comparison of the final packaged scope measures
**104.701 → 94.168 ms (1.112x)**, with 15 observations per arm and AB/BA/AB
ordering. It includes registry scope setup/cleanup on every forward. This is
distinct from cold E2E; all repetitions are exact and restore original methods.
[Profiles, kernel build pin, warm observations and rejected trials](../provenance/atlaslm_optimization_trials_20261016.json).

Use the composed request and optimized profile from the examples above with:

```text
/media/IntelSSD/fireprot/outputs/experiments/atlaslm-speed-20261016/qualified-assign/target/release/fireprot
```

`experiments/plm/prepare_atlaslm_pairs.py` reproduces the three pairs in a fresh
frozen deployment with the recorded HIP build receipt. The earlier native and
RoPE-only requests/profiles remain available. This standalone optimization does
not change AtlasFold's stochastic LM passes or its published folding speedups;
the folding model's padded LM lengths require separate qualification.

### Rejected loading and head trials

All nine diagnostic runs retained bitwise output equality. Single-process load
times, including imports inside the loader, were:

| Model | Native | Per-tensor asynchronous transfer | Packed pinned-host transfer |
| --- | ---: | ---: | ---: |
| ESMC | 10.142 s | 18.828 s | 26.475 s |
| AtlasLM | 5.280 s | 5.121 s | 6.932 s |
| ESM3 | 4.623 s | 4.620 s | 5.949 s |

The ESM3 candidate also omitted auxiliary heads not exported by the sequence
interface, saving only about 1 ms of warm forward time. These are single-trial
diagnostics, not supervised E2E measurements or statistically established gains.
None is promoted. The small AtlasLM difference does not establish an E2E gain,
and pinned-memory packing is slower for all three models.
[Trial evidence](../provenance/plm_loading_screens_20261015.json).

## ESM3 native qualification

Official ESM3-open weights were downloaded at revision
`47f0545b2b6daf26a93439a3cd610f4f7f3d5478`; both the local download and its
DiamondHill copy match official LFS SHA-256
`5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9`.
The model card is retained beside the checkpoint. [Asset record](../provenance/esm3_assets_20261015.json).

Both supported loading routes use the pinned architecture and local weights.
Nonpersistent RoPE buffers retain the native construction and casting behavior.
The full native forward still computes its heads; this entry exports sequence
logits and prenorm embeddings. No auxiliary codec is invoked by this interface.

On HIP7, N1000/seed101, all three output arrays pass Rust schema-5 acceptance and
are bitwise equal across the supervised worker, the standalone loader and the
independent official `ESM3.from_pretrained` factory with local asset routing.
That initial qualification did not establish an ESM3 acceleration ratio or validate sequence
sampling, structure decoding or multimodal prompting.
[Qualification record](../provenance/esm3_qualification_20261015.json).

### ESM3 loader E2E decision

Three subsequent AB/BA/AB N1000 pairs on HIP6 compare the official
`ESM3.from_pretrained` factory (routing only `data_root` to the verified local
checkpoint) against the previously integrated strict mmap/assign loader. Both
arms retain native Torch BF16 math, all output heads, seed101, one forward,
fresh processes and unflushed caches. All six archives and all three arrays
pass the unchanged Rust gates and bitwise comparisons.

| Loading route | Complete invocation median |
| --- | ---: |
| Official factory | **13.489 s** |
| Previous local mmap/assign loader | 14.564 s |

The mmap route is **0.926x versus the official factory**, so it is not promoted
as an optimization. Selecting the official factory instead improves the earlier
integration by 1.080x; this is not a speedup over the official implementation.
[Full observations and receipt pins](../provenance/plm_esm3_loader_e2e_20261015.json).

The prepared ESM3 profile now explicitly sets
`FIREPROT_ESM3_REFERENCE_FACTORY=1` and uses this matched CLI/worker deployment:

```text
/media/IntelSSD/fireprot/outputs/experiments/plm-esm3-load-20261015/target/release/fireprot
```

Use it with `configs/diamondhill/plm-esm3-open.json`, or build the current checkout
and deploy its workers explicitly. Setting the flag to `0` retains the earlier
strict mmap/assign loader; omission also retains that general local-loader path.
The official factory requires the verified `data/weights/esm3_sm_open_v1.pth`
filename/layout. No remote download is introduced by this flag. Reproduce with
`experiments/plm/prepare_esm3_loader_pairs.py` in a fresh frozen deployment.
Neither prepared example is a dependency installer.
