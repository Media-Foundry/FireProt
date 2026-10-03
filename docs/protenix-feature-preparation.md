# CPU feature preparation profile

## Measured scope

`profile_features.py` directly calls the native `InferenceDataset.process_one`
for the single N1536 performance input, without loading model weights or
initializing a GPU. Protein/RNA MSA, templates and ESM are disabled. Native
feature dtypes, seed 101 reset before each call, one Torch/BLAS thread, two
calls in a fresh process. Source base `5859ca2`, Protenix `fb1f39a`. Filesystem
caches were already warm from inference, while CCD process caches initialize
on the first call. Timings include cProfile overhead and should not be treated
as an uninstrumented request benchmark. CPU/GPU inference cycles, diffusion
steps and prediction samples do not apply.

| Call | Total | General features (`crop`) | MSA/template/conversion (`featurizer`) |
| --- | ---: | ---: | ---: |
| First process call | 21.879 s | 18.259 s | 3.617 s |
| Reused process caches | 6.667 s | 3.056 s | 3.608 s |

The input has 1536 tokens and 12,139 atoms. Returned tensors occupy
2,975,649,984 logical bytes. `torch.cuda.is_initialized()` remains false.
The first call spends 12.089 cumulative seconds in CCD CIF loading and
2.904 own seconds in pickle loading. The text parser executes millions of
string operations; instrumentation overhead is substantial for that path.

## Warm-call findings

- `make_template_feature`: 2.765 cumulative seconds, although template search
  is disabled. `as_protenix_dict` still computes four template distograms and
  coordinate direction arrays.
- The template tensors total 1,662,910,464 bytes. `template_distogram` alone
  has shape `[4,1536,1536,39]` and occupies 1,472,200,704 FP32 bytes.
- `dict_to_tensor`: 0.842 cumulative seconds. This includes copying feature
  arrays and is part of the broader conversion phase, so timings must not
  be summed across overlapping scopes.
- `bond_mask` has shape `[12139,12139]`, dtype int64, and occupies
  1,178,842,568 bytes. The mask-feature and dtype-conversion paths merit a
  separate consumer audit before omitting this tensor in inference.

Mini-default v0.5.0 inherits `template_embedder.n_blocks=0`. Its
`TemplateEmbedder.forward` returns zero before reading template tensors in
that configuration. This makes avoiding unused template preparation a concrete
next candidate. Template-enabled models (including the v1 Base configuration)
require their own semantics and must not inherit a blanket bypass.

No featurizer behavior is changed by this profile. A future optimization must
verify all retained features and random-state advancement, isolate supported
model/configuration cases, preserve template-enabled behavior, and pass
full-model coordinate/confidence gates plus fresh-worker timing. Persistent
CCD caching is another candidate, with resource/schema/source invalidation
requirements; it has not been implemented or timed here.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/profile_features.py \
  --input benchmarks/protenix/inputs/umem_L01536_6b140cc19f02d9b9.json \
  --output outputs/benchmarks/features-1536.json
```

[Evidence](../provenance/protenix_features_1536_5859ca2.json) retains source/input
hashes, native stage timers, tensor shapes and dtypes, RNG-state hashes and
cProfile own/cumulative leaders for both calls. Full pstats artifacts stay in
the excluded remote outputs directory.
