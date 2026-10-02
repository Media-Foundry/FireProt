"""Narrow runtime hooks that expose Protenix calls to FireProt dispatch."""

from __future__ import annotations

import functools
import os
import threading
from typing import Any


_lock = threading.RLock()
_installed = False
_enabled = False


def install() -> dict[str, Any]:
    """Install idempotent inference hooks before Protenix model construction."""
    global _enabled, _installed
    with _lock:
        if _installed:
            return manifest()
        if os.environ.get("FIREPROT_DISABLE_RECYCLE_OPS") == "1":
            _installed = True
            _enabled = False
            return manifest()

        from protenix.model.modules.primitives import Transition
        from protenix.model.triangular.layers import OpenFoldLayerNorm

        from fireprot.backends.amd.recycle import (
            dispatch_layer_norm,
            dispatch_transition,
        )

        original_layer_norm = OpenFoldLayerNorm.forward
        original_transition = Transition.forward

        @functools.wraps(original_layer_norm)
        def layer_norm_forward(self: Any, x: Any) -> Any:
            return dispatch_layer_norm(
                module=self,
                x=x,
                native=lambda **_: original_layer_norm(self, x),
            )

        @functools.wraps(original_transition)
        def transition_forward(self: Any, x: Any) -> Any:
            return dispatch_transition(
                module=self,
                x=x,
                native=lambda **_: original_transition(self, x),
            )

        OpenFoldLayerNorm.forward = layer_norm_forward
        Transition.forward = transition_forward
        _installed = True
        _enabled = True
        return manifest()


def manifest() -> dict[str, Any]:
    result = {
        "installed": _installed,
        "enabled": _enabled,
        "hooks": [
            "OpenFoldLayerNorm.forward",
            "Transition.forward",
        ]
        if _enabled
        else [],
    }
    if _enabled:
        from fireprot.backends.amd.recycle import fast_path_manifest

        result["fast_paths"] = fast_path_manifest()
    return result
