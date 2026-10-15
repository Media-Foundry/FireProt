# Standalone protein language models

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
Multiple-sequence batching and masked-token/generation tasks are not connected.

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
| `esm3-open` | Tokens, native prenorm embedding and sequence logits | BF16 embedding/logits with native BF16 autocast | Official `biohub/esm3-sm-open-v1`; N1000 native factory parity passed |

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
fireprot esm configs/requests/plm-esmc-fold2-6b-n1000-native.json \
  configs/diamondhill/plm-esmc-fold2-6b.json
fireprot esm configs/requests/plm-esm2-3b-n1000-fireprot.json \
  configs/diamondhill/plm-esm2-3b.json
fireprot atlaslm configs/requests/plm-atlaslm-3b-n1000-native.json \
  configs/diamondhill/plm-atlaslm-3b.json
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
  invariant table casts are prepared for the fixed sequence. No language model
  result or stochastic folding pass is cached.
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

Three fresh-process AB/BA/AB pairs per model completed on HIP6, N1000, using a
single frozen Rust CLI and serial execution. Native precision, one forward,
seed101 (ESMC/ESM2) or seed102 (AtlasLM), unflushed filesystem/GPU library caches;
no observations were discarded. All eighteen output archives passed Rust
acceptance. All native tokens, embeddings and available logits are bitwise equal.
The read-only collector rehashed every accepted metadata/output file and compared
all retained NPY bytes. [Versioned evidence](../provenance/plm_e2e_n1000_20261015.json).

| Model | Native → opt-in E2E median | Ratio | Decision |
| --- | ---: | ---: | --- |
| ESMC bundled 6B | 36.148 → 36.798 s | 0.982x | No cold-start E2E gain; retain native as the one-off default |
| ESM2-3B | 39.304 → 18.827 s | **2.088x** | Opt-in meta initialization plus omitted unused attention mean; exact FP32 outputs |
| AtlasLM-3B | 15.913 → 16.219 s | 0.981x | No cold-start E2E gain; retain native as the one-off default |

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
bridge. Inference examples may point to a previously used output directory;
always copy requests and change their run ID/output path before execution.

## ESM3 native qualification

Official ESM3-open weights were downloaded at revision
`47f0545b2b6daf26a93439a3cd610f4f7f3d5478`; both the local download and its
DiamondHill copy match official LFS SHA-256
`5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9`.
The model card is retained beside the checkpoint. [Asset record](../provenance/esm3_assets_20261015.json).

The maintained sequence loader uses the pinned architecture and strict local
state loading. Nonpersistent RoPE buffers are constructed on CPU as in the
official factory, then moved/cast with the model. Unused auxiliary codec callbacks
reject calls rather than downloading additional models. The full native forward
still computes its heads; this entry exports sequence logits and prenorm embeddings.

On HIP7, N1000/seed101, all three output arrays pass Rust schema-5 acceptance and
are bitwise equal across the supervised worker, the standalone loader and the
independent official `ESM3.from_pretrained` factory with local asset routing.
The comparison does not establish an ESM3 acceleration ratio or validate sequence
sampling, structure decoding or multimodal prompting.
[Qualification record](../provenance/esm3_qualification_20261015.json).

The prepared ESM3 CLI/worker deployment is separate from the frozen three-model
benchmark:

```text
/media/IntelSSD/fireprot/outputs/experiments/plm-esm3-20261015/target/release/fireprot
```

Use it with `configs/diamondhill/plm-esm3-open.json`, or build the current checkout
and deploy its workers explicitly. Neither prepared example is a dependency installer.
