"""Load the explicitly built Rust policy library; never compile during inference."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from fireprot._paths import REPOSITORY_ROOT


def _load():
    configured = os.environ.get("FIREPROT_BRIDGE_LIBRARY")
    path = (
        Path(configured)
        if configured
        else REPOSITORY_ROOT / "target/release/lib_fireprot_bridge.so"
    ).resolve()
    if not path.is_file():
        raise ImportError(
            f"FireProt Rust operator bridge is missing: {path}. "
            "Run `cargo build --release -p fireprot-bridge --locked` in the repository, "
            "or set FIREPROT_BRIDGE_LIBRARY to the built library."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("_fireprot_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FireProt Rust operator bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ABI_VERSION != 1:
        raise ImportError(f"Unsupported FireProt Rust bridge ABI: {module.ABI_VERSION}")
    return module, {"name": "fireprot-core", "bridge_abi": 1, "library": str(path), "sha256": digest}


bridge, identity = _load()
