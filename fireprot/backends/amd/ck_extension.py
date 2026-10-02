"""Lazy loader for the PyTorch ROCm Composable Kernel GEMM entry point."""

from __future__ import annotations

import hashlib
import os
import pathlib
import threading
from typing import Any


_lock = threading.RLock()
_extension: Any | None = None
_load_error: str | None = None


def _source_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "csrc/fireprot_ck_ops.cpp"


def _extension_name(torch_module: Any, source: pathlib.Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(torch_module.__version__).encode())
    digest.update(str(getattr(torch_module.version, "hip", None)).encode())
    digest.update(source.read_bytes())
    return f"fireprot_ck_ops_{digest.hexdigest()[:12]}"


def load_ck_extension(*, torch_module: Any | None = None) -> Any | None:
    """Build the direct CK wrapper once, returning None on unsupported builds."""
    global _extension, _load_error
    with _lock:
        if _extension is not None:
            return _extension
        if _load_error is not None:
            return None
        if os.environ.get("FIREPROT_DISABLE_CK_EXTENSION") == "1":
            _load_error = "disabled by FIREPROT_DISABLE_CK_EXTENSION"
            return None
        try:
            if torch_module is None:
                import torch as torch_module
            if not torch_module.cuda.is_available() or not getattr(
                torch_module.version, "hip", None
            ):
                _load_error = "PyTorch HIP is unavailable"
                return None

            from torch.utils.cpp_extension import ROCM_HOME, load

            source = _source_path()
            if not source.is_file():
                _load_error = f"missing CK extension source: {source}"
                return None
            rocm_home = pathlib.Path(
                os.environ.get("FIREPROT_ROCM_HOME") or ROCM_HOME or "/opt/rocm"
            )
            repo_root = pathlib.Path(__file__).resolve().parents[3]
            cache_root = pathlib.Path(
                os.environ.get(
                    "FIREPROT_EXTENSION_CACHE",
                    repo_root / "runtime/torch_extensions",
                )
            )
            name = _extension_name(torch_module, source)
            build_directory = cache_root / name
            build_directory.mkdir(parents=True, exist_ok=True)
            _extension = load(
                name=name,
                sources=[str(source)],
                extra_include_paths=[str(rocm_home / "include")],
                extra_cflags=["-O3", "-DUSE_ROCM", "-DUSE_ROCM_CK_GEMM"],
                build_directory=str(build_directory),
                with_cuda=True,
                verbose=os.environ.get("FIREPROT_EXTENSION_VERBOSE") == "1",
            )
        except Exception as error:  # Native fallback owns unsupported builds.
            _load_error = f"{type(error).__name__}: {error}"
            return None
        return _extension


def extension_status() -> dict[str, Any]:
    with _lock:
        return {
            "loaded": _extension is not None,
            "error": _load_error,
            "api": "ATen/native/hip/ck_gemm.h",
        }


def _reset_for_testing() -> None:
    global _extension, _load_error
    with _lock:
        _extension = None
        _load_error = None
