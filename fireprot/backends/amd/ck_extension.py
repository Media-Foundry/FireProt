"""Lazy loader for the PyTorch ROCm Composable Kernel GEMM entry point."""

from __future__ import annotations

import os
import pathlib
import threading
from typing import Any


_lock = threading.RLock()
_extension: Any | None = None
_load_error: str | None = None
_build: dict[str, Any] | None = None


def _source_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "csrc/fireprot_ck_ops.cpp"


def load_ck_extension(*, torch_module: Any | None = None) -> Any | None:
    """Build the direct CK wrapper once, returning None on unsupported builds."""
    global _extension, _load_error, _build
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

            from torch.utils.cpp_extension import ROCM_HOME
            from ._native_build import load as rust_load

            source = _source_path()
            if not source.is_file():
                _load_error = f"missing CK extension source: {source}"
                return None
            rocm_home = pathlib.Path(
                os.environ.get("FIREPROT_ROCM_HOME") or ROCM_HOME or "/opt/rocm"
            )
            _extension, _build = rust_load('fireprot_ck_ops', [source],
                cxx_flags=['-O3', '-DUSE_ROCM', '-DUSE_ROCM_CK_GEMM'], hip_flags=[],
                include_paths=[rocm_home / 'include'])
        except Exception as error:  # Native fallback owns unsupported builds.
            _load_error = f"{type(error).__name__}: {error}"
            return None
        return _extension


def extension_status() -> dict[str, Any]:
    with _lock:
        return {
            "loaded": _extension is not None,
            "error": _load_error,
            "build": _build,
            "api": "ATen/native/hip/ck_gemm.h",
        }


def _reset_for_testing() -> None:
    global _extension, _load_error, _build
    with _lock:
        _extension = None
        _load_error = None
        _build = None
