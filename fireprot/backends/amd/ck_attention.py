"""Validated exact-layout FP32 CK Tile triangle attention on gfx90a."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pathlib
import subprocess
import threading
from typing import Any

from fireprot.backends.amd import triangle_attention as ta
from fireprot.ops.registry import OperationSignature, registry

IMPLEMENTATION = "amd-gfx90a-ck-tile-attention-f32-n1000"
STRIDES = (128000, 32, 128, 1)
_enabled = (
    os.environ.get("FIREPROT_ENABLE_CK_ATTENTION", "1") == "1"
    and os.environ.get("FIREPROT_DISABLE_CK_ATTENTION") != "1"
)
_lock = threading.RLock()
_library: Any = None
_entry: Any = None
_error: str | None = None
_build: dict[str, Any] = {}


def load_extension() -> Any:
    """Build a source/header/compiler fingerprinted standalone HIP library once."""
    global _library, _entry, _error, _build
    with _lock:
        if _entry is not None or _error is not None:
            return _entry
        if os.environ.get("FIREPROT_DISABLE_CK_EXTENSION") == "1":
            _error = "disabled by FIREPROT_DISABLE_CK_EXTENSION"
            return None
        try:
            import fcntl
            import torch

            if not torch.version.hip:
                raise RuntimeError("PyTorch HIP is required")
            root = pathlib.Path(os.environ.get("FIREPROT_ROCM_HOME", "/opt/rocm"))
            version = (root / "include/ck/version.h").read_text()
            if "8d1ae90eff7d022f26019ec55b2ec6a7674b3112" not in version:
                raise RuntimeError("CK installation is outside the measured version")
            compiler = root / "bin/hipcc"
            compiler_version = subprocess.check_output([str(compiler), "--version"], text=True)
            source = pathlib.Path(__file__).parent / "csrc/fireprot_ck_attention.cpp"
            digest = hashlib.sha256()
            headers = sorted((root / "include/ck_tile").rglob("*.hpp"))
            if not headers:
                raise RuntimeError("CK Tile headers are unavailable")
            for path in headers:
                digest.update(str(path.relative_to(root / "include")).encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            header_digest = digest.hexdigest()
            for value in (source.read_bytes(), pathlib.Path(__file__).read_bytes(),
                          compiler_version.encode(), str(torch.__version__).encode(),
                          str(torch.version.hip).encode()):
                digest.update(value)
            cache = pathlib.Path(os.environ.get("FIREPROT_EXTENSION_CACHE",
                                                pathlib.Path(__file__).resolve().parents[3] / "runtime/torch_extensions"))
            directory = cache / ("ck_attention_" + digest.hexdigest()[:16])
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / "attention.so"
            command = [str(compiler), "-O3", "-std=c++17", "-fPIC", "-shared",
                       "--offload-arch=gfx90a", f"-I{root / 'include'}", str(source),
                       "-o", str(directory / "attention.pending.so")]
            with (directory / "build.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not output.is_file():
                    with (directory / "build.log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    (directory / "attention.pending.so").replace(output)
            _build = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          header_sha256=header_digest, header_count=len(headers),
                          compiler_version=compiler_version, command=command,
                          library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          library=str(output), ck_version=version)
            (directory / "manifest.json").write_text(json.dumps(_build, indent=2) + "\n")
            _library = ctypes.CDLL(str(output))
            _entry = _library.ck_attention
            _entry.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
            _entry.restype = ctypes.c_int
        except Exception as error:
            _error = f"{type(error).__name__}: {error}"
            return None
        return _entry


def extension_status() -> dict[str, Any]:
    with _lock:
        return dict(enabled=_enabled, loaded=_entry is not None, error=_error, build=dict(_build))


def _supports(signature: OperationSignature) -> bool | tuple[bool, str]:
    if not _enabled:
        return False, "CK Tile attention disabled by environment"
    if signature.shape != ta.LONG_QKV_SHAPE:
        return False, "CK Tile attention requires N1000"
    attributes = dict(signature.attributes)
    if attributes.get("bias_shapes") != repr(ta.LONG_UNMASKED_BIAS_SHAPES):
        return False, "CK Tile attention requires implicit pair mask"
    if attributes.get("qkv_strides") != repr((STRIDES,) * 3) or attributes.get("ck_tile_compatible") != "True":
        return False, "CK Tile attention requires exact QKV shape, strides, dtype, device and alignment"
    supported = ta._supports_long_direct(signature)
    if supported is not True:
        return supported
    if load_extension() is None:
        return False, "CK Tile attention extension unavailable"
    return True


def attention(*, module: Any, q: Any, k: Any, v: Any, biases: Any, native: Any) -> Any:
    import torch

    entry = load_extension()
    if entry is None:
        return ta._direct_sdpa(module=module, q=q, k=k, v=v, biases=biases, native=native)
    with torch.cuda.device(q.device):
        bias = biases[0].contiguous()
        output = torch.empty_like(q)
        status = entry(q.data_ptr(), k.data_ptr(), v.data_ptr(), bias.data_ptr(),
                       output.data_ptr(), 1000, 1000, torch.cuda.current_stream(q.device).cuda_stream)
        if status:
            raise RuntimeError(f"CK Tile attention launch failed with HIP status {status}")
    return output.transpose(-2, -3)


registry.register(operation=ta.OPERATION, name=IMPLEMENTATION, function=attention,
                  supports=_supports, priority=30, version="1")
