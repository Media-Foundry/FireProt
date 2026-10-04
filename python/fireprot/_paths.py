"""Repository resource locations for the explicit Python/tensor boundary."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = REPOSITORY_ROOT / "kernels" / "amd"
