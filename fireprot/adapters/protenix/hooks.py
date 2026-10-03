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


def _ck_ops_enabled() -> bool:
    return (
        os.environ.get("FIREPROT_ENABLE_CK_OPS") == "1"
        and os.environ.get("FIREPROT_DISABLE_CK_OPS") != "1"
    )


def install() -> dict[str, Any]:
    """Install idempotent inference hooks before Protenix model construction."""
    global _ck_enabled, _enabled, _hook_names, _installed, _recycle_enabled
    with _lock:
        if _installed:
            return manifest()
        recycle_disabled = os.environ.get("FIREPROT_DISABLE_RECYCLE_OPS") == "1"
        ck_disabled = not _ck_ops_enabled()
        attention_gate_enabled = os.environ.get("FIREPROT_ENABLE_ATTENTION_GATE_1536") == "1"
        qkv_enabled = os.environ.get("FIREPROT_ENABLE_QKV_PROJECTION") == "1"
        if recycle_disabled and ck_disabled and not attention_gate_enabled and not qkv_enabled:
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

        if attention_gate_enabled:
            from protenix.model.triangular.layers import Attention
            from fireprot.backends.amd.attention_gate import wrap_up

            original_wrap_up = Attention._wrap_up

            @functools.wraps(original_wrap_up)
            def attention_wrap_up(self: Any, o: Any, q_x: Any) -> Any:
                return wrap_up(module=self, o=o, x=q_x,
                               native=lambda: original_wrap_up(self, o, q_x))

            Attention._wrap_up = attention_wrap_up
            hook_names.append("Attention._wrap_up")

        if qkv_enabled:
            from protenix.model.triangular.layers import Attention
            from fireprot.backends.amd.qkv_projection import prepare

            original_prep_qkv = Attention._prep_qkv

            @functools.wraps(original_prep_qkv)
            def attention_prep_qkv(self: Any, q_x: Any, kv_x: Any, apply_scale: bool = True) -> Any:
                return prepare(module=self, q_x=q_x, kv_x=kv_x, apply_scale=apply_scale,
                               native=lambda: original_prep_qkv(self, q_x, kv_x, apply_scale))

            Attention._prep_qkv = attention_prep_qkv
            hook_names.append("Attention._prep_qkv")

        _hook_names = hook_names
        _installed = True
        _enabled = bool(hook_names)
        return manifest()


def manifest() -> dict[str, Any]:
    from fireprot.backends.amd.qkv_projection import manifest as qkv_manifest

    result = {
        "installed": _installed,
        "enabled": _enabled,
        "hooks": list(_hook_names),
        "qkv_projection": qkv_manifest(),
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
