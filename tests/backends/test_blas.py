from __future__ import annotations

from types import SimpleNamespace

import pytest

from fireprot.backends.amd import blas


class _Preference:
    def __init__(self) -> None:
        self.current = "_BlasBackend.Cublaslt"

    def __call__(self, value: str | None = None) -> str:
        if value is not None:
            self.current = f"_BlasBackend.{value}"
        return self.current


@pytest.fixture(autouse=True)
def _reset_policy() -> None:
    blas._reset_for_testing()
    yield
    blas._reset_for_testing()


def _fake_torch(preference: _Preference) -> SimpleNamespace:
    return SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(preferred_blas_library=preference)
        )
    )


def test_process_policy_is_idempotent_but_cannot_switch() -> None:
    preference = _Preference()
    torch = _fake_torch(preference)

    first = blas.configure_process_blas("hipblas", torch_module=torch)
    second = blas.configure_process_blas("hipblas", torch_module=torch)

    assert first == second
    assert first["effective"] == "_BlasBackend.hipblas"
    with pytest.raises(RuntimeError, match="process-global"):
        blas.configure_process_blas("ck", torch_module=torch)


def test_auto_policy_preserves_existing_preference() -> None:
    preference = _Preference()

    result = blas.configure_process_blas(
        "auto", torch_module=_fake_torch(preference)
    )

    assert result["changed"] is False
    assert result["before"] == result["effective"]


def test_shape_measurement_requires_speedup_and_no_p95_regression() -> None:
    native = blas.BlasMeasurement("hipblaslt", median_us=100, p95_us=120)
    candidates = {
        "hipblas": blas.BlasMeasurement("hipblas", median_us=80, p95_us=110),
        "ck": blas.BlasMeasurement("ck", median_us=60, p95_us=130),
    }

    result = blas.select_blas_backend(native, candidates, minimum_speedup=1.10)

    assert result["selected"] == "hipblas"
    assert result["promoted"] is True
    assert result["p95_regression"] is False


def test_unsupported_measurement_is_never_selected() -> None:
    native = blas.BlasMeasurement("hipblaslt", median_us=100, p95_us=120)
    candidates = {
        "ck": blas.BlasMeasurement(
            "ck", median_us=1, p95_us=1, supported=False, reason="unsupported"
        )
    }

    assert blas.select_blas_backend(native, candidates)["selected"] == "hipblaslt"
