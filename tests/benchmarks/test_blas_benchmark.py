from __future__ import annotations

from benchmarks.protenix.benchmark_blas import _select_summary


def _report(backend: str, median: float, p95: float, supported: bool = True):
    record = {
        "id": "float32:mm:124x128x128",
        "backend": backend,
        "supported": supported,
    }
    if supported:
        record["timing"] = {"median_us": median, "p95_us": p95}
    else:
        record["error"] = {"message": "unsupported shape"}
    return {"status": "ok", "measurements": [record]}


def test_summary_promotes_fast_candidate_without_p95_regression() -> None:
    reports = {
        "auto": _report("auto", 100, 120),
        "hipblaslt": _report("hipblaslt", 95, 115),
        "hipblas": _report("hipblas", 75, 100),
        "ck": _report("ck", 50, 140),
    }

    decisions = _select_summary(reports, minimum_speedup=1.10)

    assert decisions[0]["decision"]["selected"] == "hipblas"
    assert decisions[0]["decision"]["promoted"] is True


def test_summary_records_unsupported_candidate_and_keeps_native() -> None:
    reports = {
        "auto": _report("auto", 100, 120),
        "hipblaslt": _report("hipblaslt", 99, 110),
        "ck": _report("ck", 1, 1, supported=False),
    }

    decisions = _select_summary(reports, minimum_speedup=1.10)

    assert decisions[0]["decision"]["selected"] == "auto"
    assert decisions[0]["candidates"]["ck"]["reason"] == "unsupported shape"
