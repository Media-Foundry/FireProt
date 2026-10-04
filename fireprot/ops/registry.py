"""Tensor-side API for the Rust capability registry and native fallback."""

from __future__ import annotations

import dataclasses
import json
import threading
from collections.abc import Callable, Mapping
from typing import Any

from ._bridge import bridge, identity


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
    strides: tuple[int, ...] = ()

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
        strides: tuple[int, ...] | list[int] = (),
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
            strides=tuple(int(value) for value in strides),
        )

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["shape"] = list(self.shape)
        result["strides"] = list(self.strides)
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
        self._functions: dict[tuple[str, str], OperationImplementation] = {}
        self._policy = bridge.OperatorRegistry()
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
        if not callable(function) or not callable(supports):
            raise TypeError("function and supports must be callable")

        def capability(signature_json: str):
            return supports(OperationSignature.create(**json.loads(signature_json)))

        with self._lock:
            self._policy.register(json.dumps({
                "operation": operation,
                "name": name,
                "priority": implementation.priority,
                "version": implementation.version,
            }), capability)
            self._functions[(operation, name)] = implementation
        return implementation

    def unregister(self, operation: str, name: str) -> None:
        with self._lock:
            self._policy.unregister(operation, name)
            del self._functions[(operation, name)]

    def resolve(self, signature: OperationSignature) -> Selection:
        with self._lock:
            selected = json.loads(self._policy.resolve(json.dumps(signature.as_dict())))
            metadata = selected["implementation"]
            implementation = (
                self._functions[(metadata["operation"], metadata["name"])]
                if metadata is not None else None
            )
            return Selection(
                implementation,
                tuple((item["implementation"], item["reason"]) for item in selected["rejected"]),
                selected["cache_hit"],
            )

    def dispatch(
        self,
        signature: OperationSignature,
        fallback: ImplementationFunction,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        selection = self.resolve(signature)
        function = (
            selection.implementation.function
            if selection.implementation is not None
            else fallback
        )
        return function(*args, **kwargs)

    def manifest(self) -> dict[str, Any]:
        with self._lock:
            result = json.loads(self._policy.manifest())
        result["policy_engine"] = dict(identity)
        return result

    def clear_runtime_state(self) -> None:
        with self._lock:
            self._policy.clear_runtime_state()


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
        strides=tuple(tensor.stride()),
        layout=layout,
        mode=mode,
        mask=mask,
        inplace=inplace,
        attributes=attributes,
    )
