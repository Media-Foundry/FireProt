"""Process-start BLAS policy for PyTorch ROCm builds."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Mapping
from typing import Any


SUPPORTED_POLICIES = ("auto", "hipblaslt", "hipblas", "ck")
_lock = threading.RLock()
_configured_policy: str | None = None
_configuration: dict[str, Any] | None = None


def configure_process_blas(
    policy: str = "auto", *, torch_module: Any | None = None
) -> dict[str, Any]:
    """Select PyTorch's process-wide BLAS backend exactly once."""
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Unsupported BLAS policy {policy!r}; expected one of {SUPPORTED_POLICIES}"
        )
    global _configured_policy, _configuration
    with _lock:
        if _configured_policy is not None:
            if policy != _configured_policy:
                raise RuntimeError(
                    "BLAS policy is process-global and already configured as "
                    f"{_configured_policy!r}; refusing to switch to {policy!r}"
                )
            return dict(_configuration or {})

        if torch_module is None:
            import torch as torch_module

        preference = torch_module.backends.cuda.preferred_blas_library
        before = str(preference())
        if policy != "auto":
            preference(policy)
        after = str(preference())
        _configured_policy = policy
        _configuration = {
            "requested": policy,
            "before": before,
            "effective": after,
            "changed": before != after,
        }
        return dict(_configuration)


def configuration_manifest() -> dict[str, Any] | None:
    with _lock:
        return dict(_configuration) if _configuration is not None else None


def _reset_for_testing() -> None:
    global _configured_policy, _configuration
    with _lock:
        _configured_policy = None
        _configuration = None


@dataclasses.dataclass(frozen=True, slots=True)
class BlasMeasurement:
    backend: str
    median_us: float
    p95_us: float
    supported: bool = True
    reason: str | None = None


def select_blas_backend(
    native: BlasMeasurement,
    candidates: Mapping[str, BlasMeasurement],
    *,
    minimum_speedup: float = 1.10,
) -> dict[str, Any]:
    """Select a faster supported backend or retain the native choice."""
    eligible = [
        value
        for value in candidates.values()
        if value.supported and value.median_us > 0 and value.p95_us > 0
    ]
    promotable = [
        value
        for value in eligible
        if native.median_us / value.median_us >= minimum_speedup
        and value.p95_us <= native.p95_us
    ]
    selected = min(promotable, key=lambda value: value.median_us, default=native)
    speedup = native.median_us / selected.median_us
    p95_regression = selected.p95_us > native.p95_us
    promoted = selected.backend != native.backend
    return {
        "selected": selected.backend,
        "promoted": promoted,
        "median_speedup": speedup,
        "p95_regression": p95_regression,
        "minimum_speedup": minimum_speedup,
    }
