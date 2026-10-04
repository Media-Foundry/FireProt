# Framework continuous integration

`.github/workflows/framework.yml` checks clean Linux checkouts on pushes, pull
requests and manual dispatch. It has read-only repository permissions and does
not fetch upstream model repositories, credentials, weights or generated results.

The matrix tests the declared minimum Rust version (1.85.0) and current stable.
Both run the locked Cargo test suite, build the CLI/compiler guard/CPython bridge,
inspect every checked-in request and test registry behavior across the Python/Rust
boundary. Stable additionally checks formatting and Clippy. Toolchain versions
are printed in job logs. The checkout action is pinned to its commit.

The bridge job uses a fresh Python virtual environment containing pytest 9.0.2.
It runs both registry test modules except the test requiring an actual Torch
tensor. That test remains in the full Python suite. No Torch installation or GPU
is needed for this CI job. Rust numerical fixtures are tracked binary inputs;
their NumPy-based generator is not needed to run tests.

## Reproduce locally

On Linux with Rust, Git, a C compiler and Python 3.10+:

```bash
cargo +1.85.0 test --workspace --locked
cargo test --workspace --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo build --locked -p fireprot-cli -p fireprot-runtime -p fireprot-bridge
for request in configs/requests/*.json; do
  target/debug/fireprot inspect "$request"
done
python3 -m venv /tmp/fireprot-ci-python
/tmp/fireprot-ci-python/bin/python -m pip install pytest==9.0.2
PYTHONPATH=python PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  FIREPROT_BRIDGE_LIBRARY="$PWD/target/debug/lib_fireprot_bridge.so" \
  /tmp/fireprot-ci-python/bin/python -m pytest -q \
  tests/operators/test_registry.py tests/operators/test_rust_registry.py \
  -k 'not test_tensor_signature_records_actual_strides'
```

## Acceptance boundary

These checks cover framework contracts, process failure handling, numerical
fixtures, build/cache services and the policy bridge. They do not establish GPU
kernel compatibility, full model parity or inference performance. The ignored
hardware-monitor integration test requires the documented ROCm probe/fixture;
DiamondHill operator and complete-model checks remain separate acceptance gates.
CI passing does not complete the Rust migration or resolve OpenFold3 parity.

Deployment packaging and retirement of legacy model entry points remain pending.
The workflow does not publish binaries or claim portable model installation.
