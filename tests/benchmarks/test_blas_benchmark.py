from __future__ import annotations

import json
import pathlib

from benchmarks.protenix.benchmark_blas import (
    _extract_observed_shapes,
    _select_summary,
)


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


def test_extracts_every_valid_observed_blas_signature(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "torch-operators.json"
    source.write_text(
        json.dumps(
            [
                {
                    "key": "aten::mm",
                    "input_shapes": [[15376, 128], [128, 128]],
                    "count": 1588,
                    "self_device_time_total_us": 76558.9,
                },
                {
                    "key": "aten::bmm",
                    "input_shapes": [[16, 124, 24], [16, 24, 124]],
                    "count": 68,
                    "self_device_time_total_us": 1700.7,
                },
                {
                    "key": "aten::addmm",
                    "input_shapes": [[384], [124, 384], [384, 384], [], []],
                    "count": 68,
                    "self_device_time_total_us": 1508.1,
                },
                {
                    "key": "aten::copy_",
                    "input_shapes": [[4], [4], []],
                    "count": 999,
                    "self_device_time_total_us": 99999,
                },
                {
                    "key": "aten::mm",
                    "input_shapes": [[3, 4], [5, 6]],
                    "count": 1,
                    "self_device_time_total_us": 10,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _extract_observed_shapes(source, minimum_calls=2)

    assert result == [
        {
            "kind": "mm",
            "shape": [15376, 128, 128],
            "observed_count": 1588,
            "observed_device_time_us": 76558.9,
        },
        {
            "kind": "bmm",
            "shape": [16, 124, 24, 124],
            "observed_count": 68,
            "observed_device_time_us": 1700.7,
        },
        {
            "kind": "addmm",
            "shape": [124, 384, 384],
            "observed_count": 68,
            "observed_device_time_us": 1508.1,
        },
    ]
