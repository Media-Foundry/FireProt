"""Lazy build and loading for FireProt's small AMD extension."""

from __future__ import annotations

import hashlib
import os
import pathlib
import threading
from typing import Any


_lock = threading.RLock()
_extension: Any | None = None
_load_error: str | None = None


def _source_paths() -> tuple[pathlib.Path, ...]:
    root = pathlib.Path(__file__).resolve().parent / "csrc"
    return (root / "fireprot_amd_ops.cpp", root / "fireprot_amd_ops.cu")


def _extension_name(torch_module: Any, sources: tuple[pathlib.Path, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(str(torch_module.__version__).encode())
    digest.update(str(getattr(torch_module.version, "hip", None)).encode())
    for source in sources:
        digest.update(source.read_bytes())
    return f"fireprot_amd_ops_{digest.hexdigest()[:12]}"


def load_amd_extension(*, torch_module: Any | None = None) -> Any | None:
    """Build once into the excluded runtime cache, returning None on failure."""
    global _extension, _load_error
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

            from torch.utils.cpp_extension import ROCM_HOME, load

            sources = _source_paths()
            missing = [str(path) for path in sources if not path.is_file()]
            if missing:
                _load_error = f"missing AMD extension sources: {', '.join(missing)}"
                return None
            rocm_home = pathlib.Path(
                os.environ.get("FIREPROT_ROCM_HOME") or ROCM_HOME or "/opt/rocm"
            )
            repo_root = pathlib.Path(__file__).resolve().parents[3]
            cache_root = pathlib.Path(
                os.environ.get(
                    "FIREPROT_EXTENSION_CACHE",
                    repo_root / "runtime" / "torch_extensions",
                )
            )
            name = _extension_name(torch_module, sources)
            build_directory = cache_root / name
            build_directory.mkdir(parents=True, exist_ok=True)
            _extension = load(
                name=name,
                sources=[str(path) for path in sources],
                extra_include_paths=[str(rocm_home / "include")],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
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
        }


def _reset_for_testing() -> None:
    global _extension, _load_error
    with _lock:
        _extension = None
        _load_error = None
