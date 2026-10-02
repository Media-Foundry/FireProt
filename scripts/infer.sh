#!/usr/bin/env bash
set -euo pipefail
FIREPROT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIREPROT_PYTHON="${FIREPROT_PYTHON:-/home/pc/anaconda3/envs/fold/bin/python}"
export PROTENIX_ROOT_DIR="$FIREPROT_ROOT/runtime"
export PYTHONPATH="$FIREPROT_ROOT:$FIREPROT_ROOT/Protenix${PYTHONPATH:+:$PYTHONPATH}"
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ ! -x "$FIREPROT_PYTHON" ]]; then
  echo "Set FIREPROT_PYTHON to an existing Protenix-compatible Python. DiamondHill default: /home/pc/anaconda3/envs/fold/bin/python" >&2
  exit 2
fi
if [[ ! -f "$PROTENIX_ROOT_DIR/checkpoint/protenix_mini_esm_v0.5.0.pt" ]]; then
  echo "Weights live on DiamondHill at /media/IntelSSD/fireprot/runtime; see README.md." >&2
  exit 2
fi
FIREPROT_BACKEND="${FIREPROT_BACKEND:-native}"
FIREPROT_BLAS_BACKEND="${FIREPROT_BLAS_BACKEND:-auto}"
cd "$FIREPROT_ROOT/Protenix"
COMMON_ARGS=(
  --model_name protenix_mini_esm_v0.5.0
  --cycle 4
  --step 1
  --sample 1
  --dtype fp32
  --seeds 101
  --use_default_params false
  --use_msa false
  --use_template false
  --enable_tf32 false
)
case "$FIREPROT_BACKEND" in
  native)
    exec "$FIREPROT_PYTHON" -m runner.batch_inference \
      "${COMMON_ARGS[@]}" --trimul_kernel torch --triatt_kernel torch "$@"
    ;;
  fireprot)
    exec "$FIREPROT_PYTHON" -m fireprot.adapters.protenix.runtime \
      --fireprot-blas-backend "$FIREPROT_BLAS_BACKEND" -- \
      "${COMMON_ARGS[@]}" --trimul_kernel fireprot --triatt_kernel fireprot "$@"
    ;;
  *)
    echo "FIREPROT_BACKEND must be 'native' or 'fireprot', got '$FIREPROT_BACKEND'." >&2
    exit 2
    ;;
esac
