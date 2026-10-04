# Python boundary

The maintained framework is a Rust workspace. Python remains where pinned
upstream models and Torch tensors require it:

| Path | Responsibility |
| --- | --- |
| `fireprot_workers/` | Upstream model invocation and tensor serialization over the Rust-owned worker protocol |
| `fireprot/adapters/` | Model-specific tensor hooks, source compatibility and construction boundaries |
| `fireprot/backends/amd/` | Tensor capability checks, HIP/CK calls, Torch environment descriptors and native library loading |
| `fireprot/ops/` | Tensor metadata/callback interface to Rust registry policy |
| `fireprot/validation/` | Historical Protenix comparison helpers; maintained output acceptance is Rust-owned |

Native sources live in `kernels/amd`. The root `fireprot/__init__.py` only preserves
upstream and historical imports; it contains no implementation copy. Both the
repository root and this directory can be placed on `PYTHONPATH`.

Build the explicit bridge before importing the shared operator API:

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
PYTHONPATH=python:Protenix python -m pytest -q tests
```

Run these commands from the repository root. Model execution uses its pinned
Torch/ROCm environment and upstream checkout, not a shared generic Python install.
The Rust installer packages this boundary with the CLI, bridge and native kernel
sources; see [installation](../docs/installation.md). Model environments and
weights remain external, explicitly pinned deployment inputs. Environment
inventories cover declared roots; they do not create a hermetic runtime.
