from __future__ import annotations

import json
import pathlib
from unittest import mock

import pytest

from benchmarks.protenix import profile_inference


def test_repeated_input_labels_warmup_and_measurement(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps([{"name": "case", "sequences": [{"proteinChain": {}}]}]),
        encoding="utf-8",
    )
    destination = tmp_path / "repeated.json"

    labels = profile_inference._make_repeated_input(source, destination, 1, 2)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert [(item["phase"], item["iteration"]) for item in labels] == [
        ("warmup", 0),
        ("measure", 0),
        ("measure", 1),
    ]
    assert [item["name"] for item in payload] == [
        "case__fireprot_warmup_000",
        "case__fireprot_measure_000",
        "case__fireprot_measure_001",
    ]


def test_repeated_input_rejects_multiple_source_records(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"name": "a"}, {"name": "b"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        profile_inference._make_repeated_input(
            source, tmp_path / "repeated.json", 1, 1
        )


def test_timing_summary_separates_warmup_and_measurement() -> None:
    records = [
        {
            "phase": "warmup",
            "model_gpu_ms": 10.0,
            "model_wall_ms": 11.0,
            "peak_memory_allocated_bytes": 100,
            "peak_memory_reserved_bytes": 200,
            "stages_gpu_ms": {"pairformer": [6.0], "diffusion": [2.0]},
        },
        {
            "phase": "measure",
            "model_gpu_ms": 4.0,
            "model_wall_ms": 5.0,
            "peak_memory_allocated_bytes": 80,
            "peak_memory_reserved_bytes": 180,
            "stages_gpu_ms": {"pairformer": [2.0, 1.0], "diffusion": [0.5]},
        },
        {
            "phase": "measure",
            "model_gpu_ms": 6.0,
            "model_wall_ms": 7.0,
            "peak_memory_allocated_bytes": 120,
            "peak_memory_reserved_bytes": 220,
            "stages_gpu_ms": {"pairformer": [4.0], "diffusion": [1.5]},
        },
    ]

    summary = profile_inference._summarize_records(records)

    assert summary["warmup"]["model_gpu_ms"]["median"] == 10.0
    assert summary["measure"]["model_gpu_ms"]["median"] == 5.0
    assert summary["measure"]["stages_gpu_ms"]["pairformer"]["median"] == 3.5
    assert summary["measure"]["stages_gpu_ms"]["diffusion"]["p95"] == 1.5
    assert summary["measure"]["peak_memory_allocated_bytes"]["median"] == 100
    assert summary["measure"]["peak_memory_reserved_bytes"]["p95"] == 220


def test_exclusive_preflight_rejects_busy_selected_device() -> None:
    snapshot = {
        "returncode": 0,
        "stderr": "",
        "data": {
            "card3": {
                "GPU use (%)": "91",
                "VRAM Total Used Memory (B)": str(2 * 1024**3),
            }
        },
    }
    measurement = {
        "device_policy": "exclusive",
        "exclusive_max_gpu_use_percent": 5,
        "exclusive_max_vram_used_bytes": 256 * 1024**2,
    }

    with (
        mock.patch.object(profile_inference.shutil, "which", return_value="/bin/rocm-smi"),
        mock.patch.object(
            profile_inference, "_run_json_command", return_value=snapshot
        ),
        mock.patch.dict(
            profile_inference.os.environ,
            {"HIP_VISIBLE_DEVICES": "3"},
            clear=False,
        ),
    ):
        result = profile_inference._rocm_preflight(measurement)

    assert result["exclusive_check"] == {
        "required": True,
        "passed": False,
        "gpu_use_percent": 91,
        "vram_used_bytes": 2 * 1024**3,
        "max_gpu_use_percent": 5,
        "max_vram_used_bytes": 256 * 1024**2,
        "reason": "selected device is already active",
    }


def test_amd_smi_process_ids_selects_physical_device_and_skips_bad_rows() -> None:
    payload = [
        {
            "gpu": "6",
            "process_list": [
                {"process_info": {"pid": "123"}},
                {"process_info": {"pid": 456}},
                {"process_info": {}},
            ],
        },
        {"gpu": 7, "process_list": [{"process_info": {"pid": 789}}]},
    ]

    assert profile_inference._amd_smi_process_ids(payload, 6) == {123, 456}
    assert profile_inference._amd_smi_process_ids(payload, 7) == {789}
    assert profile_inference._amd_smi_process_ids(payload, 5) == set()


def test_gpu_process_monitor_fails_closed_when_amd_smi_is_unavailable() -> None:
    with mock.patch.object(profile_inference.shutil, "which", return_value=None):
        monitor = profile_inference._GpuProcessMonitor(6, 123, 1.0)

    monitor.start()
    report = monitor.stop()

    assert report["available"] is False
    assert report["samples"] == 0
    assert report["passed"] is False


def test_rocprof_import_groups_kernel_categories(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "kernel_stats.csv"
    source.write_text(
        "\n".join(
            [
                '"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"',
                '"vectorized_layer_norm_kernel",2,200,100,50,90,110,10',
                '"Cijk_example",4,160,40,40,35,45,5',
                '"softmax_kernel",1,40,40,10,40,40,0',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    assert profile_inference._import_rocprof(source, output) == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["kernel_count"] == 3
    assert report["dispatch_count"] == 7
    assert report["total_duration_ns"] == 400
    assert report["categories"]["layernorm"]["percentage"] == 50
    assert report["categories"]["gemm"]["percentage"] == 40
    assert report["categories"]["softmax"]["percentage"] == 10


def test_explicit_source_commits_override_synced_git_metadata(
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "case"}]', encoding="utf-8")
    config = {"measurement": {"warmups": 0, "repetitions": 1}}
    workload = {"id": "case", "input": str(input_path)}

    with (
        mock.patch.dict(
            profile_inference.os.environ,
            {
                "FIREPROT_SOURCE_COMMIT": "fireprot-explicit",
                "FIREPROT_PROTENIX_COMMIT": "protenix-explicit",
            },
        ),
        mock.patch.object(
            profile_inference, "_git_commit", return_value="remote-stale"
        ),
    ):
        spec = profile_inference._build_spec(
            config_path=config_path,
            config=config,
            workload=workload,
            backend="native",
            blas_backend="auto",
            output_root=tmp_path / "outputs",
            warmups=0,
            repetitions=1,
            profile_one=True,
            torch_profile=False,
        )

    assert spec["source"] == {
        "fireprot_commit": "fireprot-explicit",
        "protenix_commit": "protenix-explicit",
    }


def test_gpu_stage_recorder_does_not_double_count_same_label_recursion() -> None:
    class FakeEvent:
        def record(self) -> None:
            pass

    class FakeCuda:
        @staticmethod
        def Event(enable_timing: bool) -> FakeEvent:
            assert enable_timing is True
            return FakeEvent()

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})()
    recorder = profile_inference._Recorder(fake_torch, [])
    wrapped = None

    def original(depth: int) -> int:
        if depth:
            return wrapped(depth - 1)
        return 7

    wrapped = recorder.wrap_gpu_stage(original, "triangle_multiplication")

    assert wrapped(1) == 7
    assert len(recorder.gpu_events["triangle_multiplication"]) == 1


def test_model_recorder_retains_failures_swallowed_by_outer_runner() -> None:
    class FakeEvent:
        def record(self) -> None:
            pass

    class FakeCuda:
        @staticmethod
        def Event(enable_timing: bool) -> FakeEvent:
            assert enable_timing is True
            return FakeEvent()

        @staticmethod
        def reset_peak_memory_stats() -> None:
            pass

        @staticmethod
        def max_memory_allocated() -> int:
            return 123

        @staticmethod
        def max_memory_reserved() -> int:
            return 456

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})()
    recorder = profile_inference._Recorder(fake_torch, [])

    def failing_model() -> None:
        raise ValueError("bad prediction")

    with pytest.raises(ValueError, match="bad prediction"):
        recorder.wrap_model_call(failing_model)()

    assert recorder.model_failures == [
        {"call_id": 0, "type": "ValueError", "message": "bad prediction"}
    ]
    assert recorder.wall_events["model"][0]["peak_memory_allocated_bytes"] == 123
    assert recorder.wall_events["model"][0]["peak_memory_reserved_bytes"] == 456


def test_additional_gpu_monitoring_required_for_dual_attention():
    import pytest
    environment = {"HIP_VISIBLE_DEVICES": "7,6", "FIREPROT_ENABLE_TWO_GCD_ATTENTION": "1"}
    with pytest.raises(ValueError, match="monitoring both"):
        profile_inference._additional_physical_devices(environment, {})
    assert profile_inference._additional_physical_devices(environment, {"additional_physical_devices": [6]}) == [6]
    for devices in ([7], [6,6], [5]):
        with pytest.raises(ValueError):
            profile_inference._additional_physical_devices(environment, {"additional_physical_devices": devices})


def test_additional_gpu_preflight_rejects_busy_secondary(monkeypatch):
    import pytest
    monkeypatch.setattr(profile_inference, "_rocm_preflight", lambda _: dict(available=True,returncode=0,data={"card6":{"GPU use (%)":"99","VRAM Total Used Memory (B)":"0"}}))
    with pytest.raises(RuntimeError, match="GPU 6"):
        profile_inference._additional_device_preflight([6], {})
