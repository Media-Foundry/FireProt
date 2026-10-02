from __future__ import annotations

from unittest import mock

from fireprot.adapters.protenix import runtime


def test_install_configures_blas_and_skips_absent_optional_modules() -> None:
    runtime._installed = None
    with (
        mock.patch.object(
            runtime,
            "configure_process_blas",
            return_value={"requested": "hipblas", "effective": "hipblas"},
        ) as configure,
        mock.patch.object(runtime, "configuration_manifest", return_value={"x": 1}),
        mock.patch.object(runtime.importlib.util, "find_spec", return_value=None),
        mock.patch.object(runtime.registry, "manifest", return_value={"ops": []}),
    ):
        result = runtime.install("hipblas")

    configure.assert_called_once_with("hipblas")
    assert result["registration_modules"] == []
    assert result["blas"] == {"x": 1}
    assert result["operations"] == {"ops": []}


def test_parse_args_leaves_native_protenix_arguments_untouched() -> None:
    options, remaining = runtime._parse_args(
        [
            "--fireprot-blas-backend",
            "ck",
            "--",
            "--input",
            "case.json",
            "--cycle",
            "4",
        ]
    )

    assert options.fireprot_blas_backend == "ck"
    assert remaining == ["--", "--input", "case.json", "--cycle", "4"]
