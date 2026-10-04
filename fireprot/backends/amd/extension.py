"""Lazy build and loading for FireProt's small AMD extension."""

from __future__ import annotations

import os
import pathlib
import threading
from typing import Any


_lock = threading.RLock()
_extension: Any | None = None
_load_error: str | None = None
_build: dict[str, Any] | None = None


def _source_paths() -> tuple[pathlib.Path, ...]:
    root = pathlib.Path(__file__).resolve().parent / "csrc"
    return (root / "fireprot_amd_ops.cpp", root / "fireprot_amd_ops.cu")


def load_amd_extension(*, torch_module: Any | None = None) -> Any | None:
    """Build once into the excluded runtime cache, returning None on failure."""
    global _extension, _load_error, _build
    with _lock:
        if _extension is not None:
            return _extension
        if _load_error is not None:
            return None
        if os.environ.get("FIREPROT_DISABLE_AMD_EXTENSION") == "1":
            _load_error = "disabled by FIREPROT_DISABLE_AMD_EXTENSION"
            return None
        try:
            if torch_module is None:
                import torch as torch_module
            if not torch_module.cuda.is_available() or not getattr(
                torch_module.version, "hip", None
            ):
                _load_error = "PyTorch HIP is unavailable"
                return None

            from ._published import select_python
            published = select_python('fireprot_amd_ops', 'FIREPROT_AMD_OPS')
            if published is not None:
                _extension, _build = published
                return _extension

            from torch.utils.cpp_extension import ROCM_HOME
            from ._native_build import load as rust_load

            sources = _source_paths()
            missing = [str(path) for path in sources if not path.is_file()]
            if missing:
                _load_error = f"missing AMD extension sources: {', '.join(missing)}"
                return None
            rocm_home = pathlib.Path(
                os.environ.get("FIREPROT_ROCM_HOME") or ROCM_HOME or "/opt/rocm"
            )
            _extension, _build = rust_load('fireprot_amd_ops', sources, hip_flags=['-O3'],
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
        }


def _reset_for_testing() -> None:
    global _extension, _load_error, _build
    with _lock:
        _extension = None
        _load_error = None
        _build = None
