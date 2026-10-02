"""Experimental exact N1536 FP32 hipBLASLt contraction with native fallback."""
from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import subprocess
import threading
from typing import Any

_lock = threading.RLock()
_library: Any = None
_entry: Any = None
_error: str | None = None
_build: dict[str, Any] = {}
_counts = {"outgoing": 0, "incoming": 0, "fallback": 0}
WORKSPACE_BYTES = 32 * 1024 * 1024


def load_extension() -> Any:
    global _library, _entry, _error, _build
    with _lock:
        if _entry is not None or _error is not None:
            return _entry
        try:
            import fcntl
            import torch
            root = pathlib.Path(os.environ.get("FIREPROT_ROCM_HOME", "/opt/rocm"))
            compiler = root / "bin/hipcc"
            version = subprocess.check_output([str(compiler), "--version"], text=True)
            source = pathlib.Path(__file__).parent / "csrc/fireprot_lt_bmm.cpp"
            digest = hashlib.sha256(source.read_bytes() + pathlib.Path(__file__).read_bytes())
            digest.update((version + str(torch.__version__) + str(torch.version.hip)).encode())
            for header in sorted((root / "include/hipblaslt").glob("*.h")):
                digest.update(header.name.encode() + header.read_bytes())
            library = (root / "lib/libhipblaslt.so").resolve()
            stat = library.stat()
            digest.update(f"{root.resolve()}:{library}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            cache = pathlib.Path(os.environ.get("FIREPROT_EXTENSION_CACHE",
                                 pathlib.Path(__file__).resolve().parents[3] / "runtime/torch_extensions"))
            directory = cache / ("lt_bmm_" + digest.hexdigest()[:16])
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / "bmm.so"
            command = [str(compiler), "-O3", "-std=c++17", "-fPIC", "-shared",
                       "--offload-arch=gfx90a", f"-I{root / 'include'}", str(source),
                       f"-L{root / 'lib'}", f"-Wl,-rpath,{root / 'lib'}", "-lhipblaslt",
                       "-o", str(directory / "pending.so")]
            with (directory / "build.lock").open("w") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                if not output.is_file():
                    with (directory / "build.log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    (directory / "pending.so").replace(output)
            _library = ctypes.CDLL(str(output))
            _entry = _library.lt_packed_bmm
            _entry.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2 + [ctypes.c_size_t]
            _entry.restype = ctypes.c_int
            _build = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          fingerprint=digest.hexdigest(), command=command, compiler=version)
        except Exception as error:
            _error = f"{type(error).__name__}: {error}"
            return None
        return _entry


def supported(a: Any, b: Any) -> bool:
    import torch
    if os.environ.get("FIREPROT_ENABLE_LT_BMM_1536") != "1":
        return False
    if torch.is_grad_enabled() or not torch.version.hip or torch.backends.cuda.matmul.allow_tf32:
        return False
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        return False
    if any(x.dtype != torch.float32 or tuple(x.shape) != (128,1536,1536)
           or not x.is_contiguous() or x.requires_grad or x.data_ptr() % 16 for x in (a,b)):
        return False
    if torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0] != "gfx90a":
        return False
    with torch.cuda.device(a.device):
        return not torch.cuda.is_current_stream_capturing()


def packed_bmm(a: Any, b: Any, outgoing: bool) -> Any:
    import torch
    global _error
    entry = load_extension() if supported(a,b) else None
    if entry is not None:
        with torch.cuda.device(a.device):
            output = torch.empty_like(a)
            workspace = torch.empty(WORKSPACE_BYTES, device=a.device, dtype=torch.uint8)
            stream = torch.cuda.current_stream(a.device)
            status = entry(a.data_ptr(), b.data_ptr(), output.data_ptr(), 128,1536,int(outgoing),
                           stream.cuda_stream,workspace.data_ptr(),workspace.numel())
            if status == 0:
                with _lock:
                    _counts["outgoing" if outgoing else "incoming"] += 1
                return output
            # A launch error can leave asynchronous device work in a failed state.
            # Do not hide it by silently attempting a native kernel.
            raise RuntimeError(f"hipBLASLt contraction failed with status {status}")
    with _lock:
        _counts["fallback"] += 1
    return torch.bmm(a,b.transpose(-1,-2)) if outgoing else torch.bmm(a.transpose(-1,-2),b)


def extension_status() -> dict[str, Any]:
    with _lock:
        return dict(enabled=os.environ.get("FIREPROT_ENABLE_LT_BMM_1536") == "1",
                    loaded=_entry is not None, error=_error, build=dict(_build),
                    counts=dict(_counts), workspace_bytes_per_call=WORKSPACE_BYTES)
