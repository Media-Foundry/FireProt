"""FireProt bootstrap for the independent Protenix checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import pathlib
import runpy
import sys
from typing import Any

from fireprot.backends.amd.blas import (
    configuration_manifest,
    configure_process_blas,
)
from fireprot.ops.registry import registry


_OPTIONAL_REGISTRATION_MODULES = (
    "fireprot.backends.amd.ck_gemm",
    "fireprot.backends.amd.triangle_multiplication",
    "fireprot.backends.amd.triangle_attention",
    "fireprot.backends.amd.ck_attention",
    "fireprot.backends.amd.recycle",
)
_installed: dict[str, Any] | None = None


def install(blas_backend: str | None = None) -> dict[str, Any]:
    """Install FireProt runtime policy before constructing a Protenix model."""
    global _installed
    requested = blas_backend or os.environ.get("FIREPROT_BLAS_BACKEND", "auto")
    blas = configure_process_blas(requested)
    imported = []
    for module_name in _OPTIONAL_REGISTRATION_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            importlib.import_module(module_name)
            imported.append(module_name)
    from fireprot.adapters.protenix import hooks

    hook_manifest = hooks.install()
    from fireprot.adapters.protenix import construction
    construction.install()
    _installed = {
        "adapter": "protenix",
        "blas": blas,
        "registration_modules": imported,
        "hooks": hook_manifest,
    }
    return manifest()


def manifest() -> dict[str, Any]:
    from fireprot.adapters.protenix.features import manifest as features_manifest
    from fireprot.adapters.protenix.construction import manifest as construction_manifest
    from fireprot.backends.amd.extension import extension_status as amd_status
    from fireprot.backends.amd.ck_extension import extension_status as ck_gemm_status
    from fireprot.backends.amd.ck_attention import extension_status
    from fireprot.backends.amd.lt_bmm import extension_status as lt_status
    from fireprot.backends.amd.attention_gate import manifest as gate_manifest
    from fireprot.backends.amd.two_gcd_attention import manifest as two_gcd_manifest
    from fireprot.backends.amd.triangle_multiplication import vector_manifest

    result = {
        **(_installed or {"adapter": "protenix", "installed": False}),
        "blas": configuration_manifest(),
        "operations": registry.manifest(),
        "amd_extension": amd_status(),
        "ck_gemm_extension": ck_gemm_status(),
        "ck_attention": extension_status(),
        "lt_bmm": lt_status(),
        "attention_gate": gate_manifest(),
        "two_gcd_attention": two_gcd_manifest(),
        "triangle_vector": vector_manifest(),
        "construction": construction_manifest(),
        "feature_preparation": features_manifest(),
    }
    if _installed is not None:
        from fireprot.adapters.protenix import hooks

        result["hooks"] = hooks.manifest()
    return result


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fireprot-blas-backend",
        choices=("auto", "hipblaslt", "hipblas", "ck"),
        default=os.environ.get("FIREPROT_BLAS_BACKEND", "auto"),
    )
    parser.add_argument("--fireprot-manifest", type=pathlib.Path)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    options, protenix_args = _parse_args(list(argv or sys.argv[1:]))
    if protenix_args[:1] == ["--"]:
        protenix_args = protenix_args[1:]
    install(options.fireprot_blas_backend)
    sys.argv = ["runner.batch_inference", *protenix_args]
    exit_code = 0
    try:
        runpy.run_module("runner.batch_inference", run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            exit_code = int(error.code)
    finally:
        if options.fireprot_manifest is not None:
            options.fireprot_manifest.parent.mkdir(parents=True, exist_ok=True)
            options.fireprot_manifest.write_text(
                json.dumps(manifest(), indent=2) + "\n", encoding="utf-8"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
