#!/usr/bin/env bash
set -euo pipefail
FIREPROT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -a -e 'ssh -o ConnectTimeout=15' \
  --exclude='.git' --exclude='.gitnexus' --exclude='__pycache__' \
  --exclude='.ninja_*' --exclude='build.ninja' \
  --exclude='*.o' --exclude='*.so' --exclude='fireprot_amd_ops.hip' \
  --exclude='outputs' --exclude='runtime' \
  "$FIREPROT_ROOT/Protenix" "$FIREPROT_ROOT/benchmarks" \
  "$FIREPROT_ROOT/fireprot" "$FIREPROT_ROOT/tests" "$FIREPROT_ROOT/cases" \
  "$FIREPROT_ROOT/scripts" "$FIREPROT_ROOT/docs" "$FIREPROT_ROOT/README.md" \
  pc@10.120.16.9:/media/IntelSSD/fireprot/
