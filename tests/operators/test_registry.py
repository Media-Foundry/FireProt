from __future__ import annotations

from fireprot.ops.registry import OperationRegistry, OperationSignature


def _signature(**overrides: object) -> OperationSignature:
    values = {
        "operation": "triangle_multiplication",
        "device_type": "cuda",
        "architecture": "gfx90a",
        "dtype": "float32",
        "shape": (124, 124, 128),
        "layout": "contiguous",
        "mode": "inference",
        "mask": "dense",
        "inplace": True,
    }
    values.update(overrides)
    return OperationSignature.create(**values)


def test_registry_selects_highest_priority_supported_candidate() -> None:
    registry = OperationRegistry()
    registry.register(
        operation="triangle_multiplication",
        name="slow",
        function=lambda value: f"slow:{value}",
        supports=lambda signature: True,
        priority=1,
    )
    registry.register(
        operation="triangle_multiplication",
        name="fast",
        function=lambda value: f"fast:{value}",
        supports=lambda signature: signature.architecture == "gfx90a",
        priority=10,
    )

    assert registry.dispatch(_signature(), lambda value: f"native:{value}", "x") == "fast:x"
    manifest = registry.manifest()
    assert manifest["selection_counts"] == [
        {
            "operation": "triangle_multiplication",
            "implementation": "fast",
            "count": 1,
        }
    ]


def test_registry_falls_back_and_records_rejection_reason() -> None:
    registry = OperationRegistry()
    registry.register(
        operation="triangle_multiplication",
        name="gfx90a-only",
        function=lambda value: value,
        supports=lambda signature: (False, "architecture mismatch"),
    )

    result = registry.dispatch(_signature(architecture="gfx942"), lambda: "native")

    assert result == "native"
    decision = registry.manifest()["decisions"][0]
    assert decision["selected"] == "native"
    assert decision["rejected"] == [
        {
            "implementation": "gfx90a-only",
            "reason": "architecture mismatch",
        }
    ]


def test_registry_cache_is_invalidated_by_new_registration() -> None:
    registry = OperationRegistry()
    signature = _signature()

    assert registry.resolve(signature).implementation is None
    registry.register(
        operation="triangle_multiplication",
        name="new",
        function=lambda: None,
        supports=lambda candidate: True,
    )

    selection = registry.resolve(signature)
    assert selection.cache_hit is False
    assert selection.implementation is not None
    assert selection.implementation.name == "new"


def test_capability_error_fails_closed_to_native() -> None:
    registry = OperationRegistry()

    def broken_capability(signature: OperationSignature) -> bool:
        raise RuntimeError("probe failed")

    registry.register(
        operation="triangle_multiplication",
        name="broken",
        function=lambda: "wrong",
        supports=broken_capability,
    )

    assert registry.dispatch(_signature(), lambda: "native") == "native"
    reason = registry.manifest()["decisions"][0]["rejected"][0]["reason"]
    assert reason == "capability error: RuntimeError: probe failed"
