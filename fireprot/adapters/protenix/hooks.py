"""Narrow runtime hooks that expose Protenix calls to FireProt dispatch."""

from __future__ import annotations

import functools
import os
import threading
from typing import Any


_lock = threading.RLock()
_installed = False
_enabled = False
_recycle_enabled = False
_ck_enabled = False
_hook_names: list[str] = []


def install() -> dict[str, Any]:
    """Install idempotent inference hooks before Protenix model construction."""
    global _ck_enabled, _enabled, _hook_names, _installed, _recycle_enabled
    with _lock:
        if _installed:
            return manifest()
        recycle_disabled = os.environ.get("FIREPROT_DISABLE_RECYCLE_OPS") == "1"
        ck_disabled = os.environ.get("FIREPROT_DISABLE_CK_OPS") == "1"
        if recycle_disabled and ck_disabled:
            _installed = True
            _enabled = False
            return manifest()

        hook_names = []
        if not recycle_disabled:
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
            hook_names.extend(
                ["OpenFoldLayerNorm.forward", "Transition.forward"]
            )
            _recycle_enabled = True

        if not ck_disabled:
            from protenix.model.triangular.layers import OuterProductMean

            from fireprot.backends.amd.ck_gemm import (
                dispatch_ck_outer_product_mean,
            )

            original_outer_product_mean = OuterProductMean._forward

            @functools.wraps(original_outer_product_mean)
            def outer_product_mean_forward(
                self: Any,
                m: Any,
                mask: Any = None,
                chunk_size: int | None = None,
                inplace_safe: bool = False,
            ) -> Any:
                return dispatch_ck_outer_product_mean(
                    module=self,
                    m=m,
                    mask=mask,
                    chunk_size=chunk_size,
                    inplace_safe=inplace_safe,
                    native=lambda: original_outer_product_mean(
                        self, m, mask, chunk_size, inplace_safe
                    ),
                )

            OuterProductMean._forward = outer_product_mean_forward
            hook_names.append("OuterProductMean._forward")
            _ck_enabled = True

        _hook_names = hook_names
        _installed = True
        _enabled = bool(hook_names)
        return manifest()


def manifest() -> dict[str, Any]:
    result = {
        "installed": _installed,
        "enabled": _enabled,
        "hooks": list(_hook_names),
    }
    if _enabled:
        selection_counts = []
        backends = {}
        if _recycle_enabled:
            from fireprot.backends.amd.recycle import fast_path_manifest

            recycle_manifest = fast_path_manifest()
            selection_counts.extend(recycle_manifest["selection_counts"])
            backends["recycle"] = recycle_manifest
        if _ck_enabled:
            from fireprot.backends.amd.ck_gemm import fast_path_manifest

            ck_manifest = fast_path_manifest()
            selection_counts.extend(ck_manifest["selection_counts"])
            backends["ck"] = ck_manifest

        result["fast_paths"] = {
            "selection_counts": sorted(
                selection_counts,
                key=lambda item: (item["operation"], item["implementation"]),
            ),
            "backends": backends,
        }
    return result
