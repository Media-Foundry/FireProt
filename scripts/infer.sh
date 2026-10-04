#!/usr/bin/env bash
set -euo pipefail
FIREPROT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIREPROT_EXECUTABLE="${FIREPROT_BIN:-$FIREPROT_ROOT/target/release/fireprot}"
if [[ ! -x "$FIREPROT_EXECUTABLE" ]]; then
  echo "Build fireprot-cli or set FIREPROT_BIN to the installed Rust executable." >&2
  exit 127
fi
exec "$FIREPROT_EXECUTABLE" infer "$@"
