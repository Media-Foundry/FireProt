#!/usr/bin/env bash
set -euo pipefail
FIREPROT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sync_options=()
case "${1:-}" in
  --dry-run) sync_options+=(--dry-run --itemize-changes) ;;
  "") ;;
  *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
rsync -a -e 'ssh -o ConnectTimeout=15' \
  "${sync_options[@]}" \
  --exclude='.git' --exclude='.gitnexus' --exclude='__pycache__' \
  --exclude='.ninja_*' --exclude='build.ninja' \
  --exclude='*.o' --exclude='*.so' --exclude='fireprot_amd_ops.hip' \
  --exclude='outputs' --exclude='runtime' --exclude='target' \
  "$FIREPROT_ROOT/Protenix" "$FIREPROT_ROOT/benchmarks" \
  "$FIREPROT_ROOT/fireprot" "$FIREPROT_ROOT/tests" "$FIREPROT_ROOT/cases" \
  "$FIREPROT_ROOT/scripts" "$FIREPROT_ROOT/docs" "$FIREPROT_ROOT/README.md" \
  "$FIREPROT_ROOT/Cargo.toml" "$FIREPROT_ROOT/Cargo.lock" \
  "$FIREPROT_ROOT/crates" "$FIREPROT_ROOT/python" "$FIREPROT_ROOT/kernels" \
  "$FIREPROT_ROOT/configs" "$FIREPROT_ROOT/provenance" \
  pc@10.120.16.5:/media/IntelSSD/fireprot/

# Keep the explicitly archived entry points available for historical reproduction.
rsync -a -e 'ssh -o ConnectTimeout=15' \
  "${sync_options[@]}" \
  "$FIREPROT_ROOT/.agents/legacy_protenix_infer.sh" \
  "$FIREPROT_ROOT/.agents/legacy_smoke.py" \
  pc@10.120.16.5:/media/IntelSSD/fireprot/.agents/
