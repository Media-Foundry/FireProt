# Direct strided C128 normalization

This experiment removes the contiguous-input copy from three observed OpenFold3
normalization layouts. It keeps the qualified Welford reduction and affine
expressions unchanged. Contiguous calls retain `norm.cpp`: the strided kernel is
slower on those inputs.

## Files and reproduction

- `norm_strided.cpp`: standalone HIP C ABI; only the input address calculation
  differs from the qualified contiguous kernel. Output remains contiguous.
- `build_strided.py`: fingerprinted Rust HIP build service.
- `screen_strided.py`: operator comparisons against native Torch, followed by
  allocation-inclusive timing against copy plus the qualified contiguous HIP op.
- `model_scope_strided.py`: experimental full-model scope with three explicit
  shape/stride signatures, library hashes and the existing strict FP32 guards.

On the prepared DiamondHill environment, after other timed runs have ended:

```bash
HIP_VISIBLE_DEVICES='' python -m experiments.openfold3_norm.build_strided /fresh/build
HIP_VISIBLE_DEVICES=7 python -m experiments.openfold3_norm.screen_strided \
  /fresh/build/build.json /qualified/contiguous-norm.so \
  a06828d77438cd417298f19221dca4cff371ed81ba4df041283eb2ab8ebbf03a \
  /media/IntelSSD/fireprot/outputs/triatt-profile-n1536-v1 /fresh/screen
```

The frozen screen used a local copy of the existing normalization test helpers.
The repository entry point imports those same helpers from `screen.py`; both
package imports pass a CPU-only smoke check. The HIP kernel bytes and arithmetic
are unchanged. The hashed raw archive retains the exact executed source and
compiler receipt.

## Operator results

All 25 cases pass output, mean and reciprocal-standard-deviation **bitwise**
comparisons against native Torch. Coverage includes tails, nonzero storage
offsets, four affine branches, signed zeros, and both pinned trained triangle
attention captures. Strict deterministic execution and memory filling remain on.

Representative medians from the trained start-attention capture:

| Layout | Copy + contiguous HIP | Direct strided HIP |
| --- | ---: | ---: |
| Dense N1536 pair | 2.700 ms | 2.859 ms |
| Dense 256-row chunk | 0.465 ms | 0.500 ms |
| 256-column slice | 0.909 ms | 0.508 ms |
| Full transpose | 11.998 ms | 10.199 ms |
| Transposed 256-row chunk | 2.046 ms | 1.779 ms |
| Transposed narrower input | 2.021 ms | 1.728 ms |

Timing includes allocation and eight alternating observations per implementation.
Both trained captures show slower dense cases, which are excluded from the
model treatment. Evidence: `provenance/openfold3_norm_strided_operator.json`.

An initial scheduling wrapper rejected AMD SMI's explicit idle string before
building or running a GPU kernel. The corrected wrapper accepts that exact idle
representation; both receipts are retained. This was not a numerical failure.

## Full-model qualification and remaining work

OpenBind-0 N1536, FP32, R3/four trunk passes, D200/S1, seed101 and HIP7 passes all
11 arrays bitwise and full CIF bytes against the preceding qualified HIP
normalization output. Actual counts are 11,202 HIP normalization calls, 7,846
native fallbacks, 4,730 direct strided calls and **zero input copies**.

A single uninstrumented qualification observation measured 257.121 seconds model
forward and 360.361 seconds complete invocation. These numbers are **not a paired
speedup**. Evidence: `provenance/openfold3_norm_strided_model_n1536.json`.

## Paired E2E: no improvement; not promoted

Three AB/BA/AB pairs completed under
`/media/IntelSSD/fireprot/outputs/experiments/openfold3-norm-strided-pairs-20261007`.
Both arms share a frozen worker and retain all preceding optimizations. Only
`FIREPROT_NORM_STRIDED` and the profile digest differ.

| Scope | Copy + contiguous HIP median | Direct strided HIP median | Ratio |
| --- | ---: | ---: | ---: |
| Complete invocation E2E | 365.069 s | 368.236 s | 0.9914x |
| Model forward | 261.807 s | 264.061 s | 0.9915x |

E2E is 0.87% slower. Individual paired E2E ratios are 0.9974x, 0.9986x and
0.9898x; all three candidates are slower. These observations establish no
measured E2E benefit from this implementation; they do not isolate the cause or
prove a general limit for strided normalization.

All six outputs pass all 11 arrays bitwise and full CIF bytes. Each candidate
removes all 4,730 copies; each baseline retains them. Scientific settings and
strict deterministic filling remain unchanged. Processes are fresh, filesystem
caches are unflushed and model profiling is disabled. Evidence:
`provenance/openfold3_norm_strided_e2e_n1536.json`.

**Do not promote this path.** Retain the preceding HIP normalization combination
and profile its remaining hotspots. The isolated kernel and evidence remain for
reproduction. No maintained profile enables this hook. Other shapes, models and
devices have not been qualified by these results.
