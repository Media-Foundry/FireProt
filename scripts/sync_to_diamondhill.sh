#!/usr/bin/env bash
set -euo pipefail
FIREPROT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -a -e 'ssh -o ConnectTimeout=15' --exclude='.git' --exclude='__pycache__' \
  "$FIREPROT_ROOT/Protenix" "$FIREPROT_ROOT/cases" "$FIREPROT_ROOT/scripts" \
  "$FIREPROT_ROOT/README.md" pc@10.120.16.9:/media/IntelSSD/fireprot/
