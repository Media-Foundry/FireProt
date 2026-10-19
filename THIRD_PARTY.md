# Component licensing

FireProt is a multi-license distribution. The original FireProt material uses
GPL-2.0-only with the limited OpenFold 3 exception and the AMD alternative grant
in [LICENSE](LICENSE). This is not a blanket Apache-2.0 alternative for everyone.
Third-party copyrights and licenses remain unchanged.

## OpenFold 3: Apache-2.0

Upstream: <https://github.com/aqlaboratory/openfold-3>, audited revision
`7de748b7bc93adb5af0a4032d5e9f208b6e1325f`.

The following files adapt OpenFold 3 arithmetic and are licensed in their
entirety, including FireProt modifications, under
[Apache License 2.0](licenses/Apache-2.0.txt):

| FireProt file | Upstream source below `openfold3/core/model/` |
| --- | --- |
| `python/fireprot/adapters/openfold3/attention.py` | `primitives/attention.py` |
| `experiments/hip_softmax/head16-adapter.patch` | Patch to the preceding FireProt adapter; retains its Apache-2.0 terms |
| `experiments/hip_softmax/head16-control.patch` | Experimental on/off control for that same Apache-2.0 adapter |
| `experiments/hip_softmax/worker_overlay.py` | `primitives/attention.py` |
| `experiments/openfold3_diffusion/global_softmax.py` | `primitives/attention.py`; native QK/PV with grouped HIP softmax |
| `experiments/conditioning_reuse/verified_reuse.py` | `layers/diffusion_conditioning.py` |
| `experiments/conditioning_reuse/verified_atom_reuse.py` | `layers/sequence_local_atom_attention.py` |
| `experiments/conditioning_reuse/atom_cache.py` | `layers/sequence_local_atom_attention.py` |
| `experiments/triangle_memory/replay.py` | Source-pinned experimental substitutions in `layers/triangular_multiplicative_update.py`; preserves the upstream arithmetic |
| `experiments/triangle_memory/gfx1100_copy.py` | Guarded gfx1100/N1000 substitutions in the same source-pinned triangle method; preserves native arithmetic |

The file headers retain the applicable upstream copyright holders and describe
the modifications. The OpenFold 3 exception permits combination with original
FireProt GPL material; it does not relicense these Apache-2.0 files under GPLv2.
Other FireProt modules that merely invoke an upstream API remain subject to the
original-material terms unless their own notices say otherwise.

## AMD Composable Kernel: MIT

The CK patches under `patches/ck/` retain their
[original MIT notice](patches/ck/LICENSE), also reproduced in
[licenses/CK-MIT.txt](licenses/CK-MIT.txt) for installed distributions.
External CK headers retain their upstream notices. The FireProt AMD alternative
grant does not replace CK's existing MIT license.

## PyTorch: BSD-style notices

The handwritten FP32 softmax follows PyTorch's persistent-softmax evaluation
order. The experimental C128 normalization in `experiments/openfold3_norm` follows
the corresponding Welford evaluation order. The
[PyTorch copyright and BSD notice](licenses/PyTorch-BSD.txt) is retained
alongside that attribution. The kernels are handwritten FireProt code; referencing
the evaluation order does not relicense PyTorch itself. Installed PyTorch and
its bundled libraries carry additional notices in their own distributions.

## External models, libraries, weights and data

The independent `Protenix/` checkout, external OpenFold 3/AtlasFold/AtlasLM and
other model repositories, Torch/ROCm libraries, Rust dependencies, checkpoints,
CCD resources, datasets and externally supplied documents keep their respective
licenses. FireProt's license grants no new rights in these resources. They are
not made GPLv2 or covered by the AMD alternative merely by being used with
FireProt. The OpenFold 3 exception does not create a general exception for their
other Apache-2.0 dependencies.

Distributors must inspect the licenses and notices of the particular components
they include. This inventory identifies material and relevant notices in this
repository; it is not a claim that every possible dependency combination has
been legally cleared.
