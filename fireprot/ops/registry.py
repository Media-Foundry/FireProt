"""Model-neutral operation registry with capability-gated native fallback."""

from __future__ import annotations

import dataclasses
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any


Capability = Callable[["OperationSignature"], bool | tuple[bool, str]]
ImplementationFunction = Callable[..., Any]


@dataclasses.dataclass(frozen=True, slots=True)
class OperationSignature:
    operation: str
    device_type: str
    architecture: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    mode: str
    mask: str
    inplace: bool
    attributes: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        operation: str,
        device_type: str,
        architecture: str,
        dtype: str,
        shape: tuple[int, ...] | list[int],
        layout: str = "contiguous",
        mode: str = "inference",
        mask: str = "none",
        inplace: bool = False,
        attributes: Mapping[str, Any] | None = None,
    ) -> "OperationSignature":
        normalized = tuple(
            sorted((str(key), str(value)) for key, value in (attributes or {}).items())
        )
        return cls(
            operation=operation,
            device_type=device_type,
            architecture=architecture,
            dtype=dtype,
            shape=tuple(int(value) for value in shape),
            layout=layout,
            mode=mode,
            mask=mask,
            inplace=bool(inplace),
            attributes=normalized,
        )

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["shape"] = list(self.shape)
        result["attributes"] = dict(self.attributes)
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class OperationImplementation:
    operation: str
    name: str
    function: ImplementationFunction
    supports: Capability
    priority: int
    version: str


@dataclasses.dataclass(frozen=True, slots=True)
class Selection:
    implementation: OperationImplementation | None
    rejected: tuple[tuple[str, str], ...]
    cache_hit: bool


class OperationRegistry:
    """Thread-safe registry whose missing match always means native fallback."""

    def __init__(self) -> None:
        self._implementations: dict[str, list[OperationImplementation]] = {}
        self._cache: dict[OperationSignature, Selection] = {}
        self._selection_counts: Counter[tuple[str, str]] = Counter()
        self._last_decisions: dict[OperationSignature, Selection] = {}
        self._lock = threading.RLock()

    def register(
        self,
        *,
        operation: str,
        name: str,
        function: ImplementationFunction,
        supports: Capability,
        priority: int = 0,
        version: str = "1",
    ) -> OperationImplementation:
        implementation = OperationImplementation(
            operation=operation,
            name=name,
            function=function,
            supports=supports,
            priority=int(priority),
            version=str(version),
        )
        with self._lock:
            existing = self._implementations.setdefault(operation, [])
            if any(item.name == name for item in existing):
                raise ValueError(f"Implementation already registered: {operation}/{name}")
            existing.append(implementation)
            existing.sort(key=lambda item: (-item.priority, item.name, item.version))
            self._cache.clear()
        return implementation

    def unregister(self, operation: str, name: str) -> None:
        with self._lock:
            existing = self._implementations.get(operation, [])
            kept = [item for item in existing if item.name != name]
            if len(kept) == len(existing):
                raise KeyError(f"Unknown implementation: {operation}/{name}")
            self._implementations[operation] = kept
            self._cache.clear()

    def resolve(self, signature: OperationSignature) -> Selection:
        with self._lock:
            cached = self._cache.get(signature)
            if cached is not None:
                result = Selection(cached.implementation, cached.rejected, True)
                self._record(signature, result)
                return result
            candidates = tuple(self._implementations.get(signature.operation, ()))

        rejected: list[tuple[str, str]] = []
        selected = None
        for candidate in candidates:
            try:
                capability = candidate.supports(signature)
                if isinstance(capability, tuple):
                    supported, reason = capability
                else:
                    supported, reason = capability, "capability returned false"
            except Exception as error:
                supported = False
                reason = f"capability error: {type(error).__name__}: {error}"
            if supported:
                selected = candidate
                break
            rejected.append((candidate.name, reason))

        result = Selection(selected, tuple(rejected), False)
        with self._lock:
            self._cache[signature] = result
            self._record(signature, result)
        return result

    def dispatch(
        self,
        signature: OperationSignature,
        native: ImplementationFunction,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        selection = self.resolve(signature)
        function = (
            selection.implementation.function
            if selection.implementation is not None
            else native
        )
        return function(*args, **kwargs)

    def _record(self, signature: OperationSignature, selection: Selection) -> None:
        name = (
            selection.implementation.name
            if selection.implementation is not None
            else "native"
        )
        self._selection_counts[(signature.operation, name)] += 1
        self._last_decisions[signature] = selection

    def manifest(self) -> dict[str, Any]:
        with self._lock:
            implementations = {
                operation: [
                    {
                        "name": item.name,
                        "priority": item.priority,
                        "version": item.version,
                    }
                    for item in values
                ]
                for operation, values in sorted(self._implementations.items())
            }
            counts = [
                {"operation": operation, "implementation": name, "count": count}
                for (operation, name), count in sorted(self._selection_counts.items())
            ]
            decisions = [
                {
                    "signature": signature.as_dict(),
                    "selected": (
                        selection.implementation.name
                        if selection.implementation is not None
                        else "native"
                    ),
                    "rejected": [
                        {"implementation": name, "reason": reason}
                        for name, reason in selection.rejected
                    ],
                }
                for signature, selection in sorted(
                    self._last_decisions.items(),
                    key=lambda item: repr(item[0]),
                )
            ]
        return {
            "implementations": implementations,
            "selection_counts": counts,
            "decisions": decisions,
        }

    def clear_runtime_state(self) -> None:
        with self._lock:
            self._cache.clear()
            self._selection_counts.clear()
            self._last_decisions.clear()


registry = OperationRegistry()


def signature_from_tensor(
    operation: str,
    tensor: Any,
    *,
    architecture: str,
    mode: str,
    mask: str,
    inplace: bool,
    attributes: Mapping[str, Any] | None = None,
) -> OperationSignature:
    layout = "contiguous" if tensor.is_contiguous() else "strided"
    return OperationSignature.create(
        operation=operation,
        device_type=tensor.device.type,
        architecture=architecture,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        layout=layout,
        mode=mode,
        mask=mask,
        inplace=inplace,
        attributes=attributes,
    )
