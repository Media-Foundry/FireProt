"""Opt-in checkpoint-covered Mini construction with native recovery and exact RNG advancement."""
from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import os
from pathlib import Path
import random
import threading
import time
from typing import Any

_lock = threading.RLock()
_installed = False
_unavailable: str | None = None
_contract: dict[str, Any] = {}
_counts = {"attempted": 0, "selected": 0, "recovered": 0, "native": 0, "errors": 0}
_reasons: dict[str, int] = {}
_records: list[dict[str, Any]] = []
_PENDING = "_fireprot_pending_construction"
_MODEL_SHA256 = "18c2a6eb5f6202ce4d708aedfbef1b71d77359fb0a9a6e7ae775330766374c28"
_HASHES = {
    "initializer": "72e6ede89fb399efd49db04631802f05cce93eb6f81ee5cf16f11f5d55c1e14d",
    "init_model": "14b306c00c2427536642f80f85a09f2a15b14b872e825614a186671b802d3c4a",
    "load_checkpoint": "6da10f959743bb9ef95d4184f64b1def38df3b9874c02ec19021e1de991316ff",
    "sampler": "bd951a20257379d57e1b9fd97844c7fae6eb6d1f0207f39572f5687fbbcd919a",
}


def _enabled() -> bool:
    return os.environ.get("FIREPROT_ENABLE_FAST_MODEL_INIT") == "1"


def _model_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0"
                      + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _capture_rng(device: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    device = torch.device(device)
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                device=device, cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], state["device"])


@contextlib.contextmanager
def _replay_rng(before: dict[str, Any]):
    current = _capture_rng(before["device"])
    try:
        _restore_rng(before)
        yield
    finally:
        _restore_rng(current)


def _reason(runner: Any, layers: Any, primitives: Any, initializer: Any, sampler: Any) -> str | None:
    import numpy as np
    if not _enabled():
        return "disabled"
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        return "requires_single_python_thread"
    if getattr(runner.configs, "model_name", None) != "protenix_mini_default_v0.5.0":
        return "unsupported_model"
    if getattr(runner.configs, "dtype", None) != "fp32":
        return "unsupported_dtype"
    if layers.trunc_normal_init_ is not initializer or primitives.trunc_normal_init_ is not initializer:
        return "initializer_rebound"
    if sampler.random_state is not np.random.mtrand._rand:
        return "custom_numpy_generator"
    return None


@contextlib.contextmanager
def _defer_initializers(layers: Any, primitives: Any, sampler: Any, statistics: dict[str, Any]):
    import torch
    original_layers = layers.trunc_normal_init_
    original_primitives = primitives.trunc_normal_init_
    owner = threading.get_ident()

    @functools.wraps(original_layers)
    def initialize(weights: Any, scale: float = 1.0, fan: str = "fan_in") -> None:
        if (threading.get_ident() != owner or not isinstance(weights, torch.nn.Parameter)
                or weights.device.type != "cpu" or weights.dtype != torch.float32
                or weights.ndim != 2 or weights.numel() == 0 or scale not in (1.0, 2.0)
                or fan != "fan_in"):
            statistics["native_initializer_calls"] += 1
            return original_layers(weights, scale=scale, fan=fan)
        sampler.random_state.uniform(size=weights.numel())
        statistics["uniform_calls"] += 1
        statistics["uniform_elements"] += weights.numel()

    layers.trunc_normal_init_ = initialize
    primitives.trunc_normal_init_ = initialize
    try:
        yield
    finally:
        layers.trunc_normal_init_ = original_layers
        primitives.trunc_normal_init_ = original_primitives


def _patch_runner(runner_class: Any, layers: Any, primitives: Any, sampler: Any) -> None:
    native_init = runner_class.init_model
    native_load = runner_class.load_checkpoint
    initializer = layers.trunc_normal_init_

    @functools.wraps(native_init)
    def init_model(self: Any) -> Any:
        with _lock:
            reason = _reason(self, layers, primitives, initializer, sampler)
            if reason is not None:
                _counts["native"] += 1
                _reasons[reason] = _reasons.get(reason, 0) + 1
                return native_init(self)
            started = time.perf_counter()
            before = _capture_rng(self.device)
            statistics = dict(uniform_calls=0, uniform_elements=0, native_initializer_calls=0)
            _counts["attempted"] += 1
            try:
                with _defer_initializers(layers, primitives, sampler, statistics):
                    result = native_init(self)
            except Exception:
                _counts["errors"] += 1
                raise
            statistics["initialization_ms"] = (time.perf_counter() - started) * 1000
            setattr(self, _PENDING, dict(before=before, statistics=statistics))
            return result

    @functools.wraps(native_load)
    def load_checkpoint(self: Any) -> Any:
        with _lock:
            pending = getattr(self, _PENDING, None)
            if pending is None:
                return native_load(self)
            delattr(self, _PENDING)
            model = self.model
            had_override = "load_state_dict" in model.__dict__
            override = model.__dict__.get("load_state_dict")
            native_state_load = model.load_state_dict
            calls = []
            missing = []
            known_coverage = True

            def observe(*args: Any, **kwargs: Any) -> Any:
                nonlocal known_coverage
                result = native_state_load(*args, **kwargs)
                calls.append((args, kwargs))
                if not hasattr(result, "missing_keys"):
                    known_coverage = False
                else:
                    missing.extend(result.missing_keys)
                return result

            started = time.perf_counter()
            record = pending["statistics"]
            try:
                model.load_state_dict = observe
                try:
                    result = native_load(self)
                finally:
                    if had_override:
                        model.load_state_dict = override
                    else:
                        del model.load_state_dict
                if not calls:
                    raise RuntimeError("Cannot verify checkpoint coverage: no state-dict load observed")
                if missing or not known_coverage:
                    # Rebuild all native initializers, including constructor-derived state.
                    # Reuse the already deserialized checkpoint; do not read it a second time.
                    training = self.model.training
                    with _replay_rng(pending["before"]):
                        native_init(self)
                        for args, kwargs in calls:
                            self.model.load_state_dict(*args, **kwargs)
                        self.model.train(training)
                    _counts["recovered"] += 1
                    record.update(selected=False, fallback_reason="incomplete_checkpoint",
                                  missing_keys=sorted(set(missing)))
                else:
                    _counts["selected"] += 1
                    record.update(selected=True, missing_keys=[])
                record["checkpoint_and_recovery_ms"] = (time.perf_counter() - started) * 1000
                if len(_records) < 16:
                    _records.append(record)
                return result
            except Exception:
                _counts["errors"] += 1
                raise

    runner_class.init_model = init_model
    runner_class.load_checkpoint = load_checkpoint


def install() -> dict[str, Any]:
    global _installed, _unavailable, _contract
    with _lock:
        if not _enabled() or _installed or _unavailable is not None:
            return manifest()
        try:
            import scipy
            from scipy.stats import truncnorm
            from scipy.stats._distn_infrastructure import rv_continuous
            from runner.inference import InferenceRunner
            from protenix.model.triangular import layers
            from protenix.model.modules import primitives

            functions = dict(initializer=layers.trunc_normal_init_, init_model=InferenceRunner.init_model,
                             load_checkpoint=InferenceRunner.load_checkpoint, sampler=truncnorm._rvs)
            hashes = {name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                      for name, fn in functions.items()}
            model_hash = _model_digest(Path(layers.__file__).resolve().parents[1])
            _contract = dict(scipy=scipy.__version__, source_sha256=hashes,
                             model_sha256=model_hash)
            if (scipy.__version__ != "1.18.1" or hashes != _HASHES
                    or model_hash != _MODEL_SHA256
                    or truncnorm._rvs.__func__ is not rv_continuous._rvs
                    or primitives.trunc_normal_init_ is not layers.trunc_normal_init_):
                _unavailable = "Unverified Protenix or SciPy construction contract"
                return manifest()
            _patch_runner(InferenceRunner, layers, primitives, truncnorm)
            _installed = True
        except Exception as error:
            _unavailable = f"{type(error).__name__}: {error}"
        return manifest()


def manifest() -> dict[str, Any]:
    with _lock:
        return dict(enabled=_enabled(), installed=_installed, unavailable=_unavailable,
                    contract=dict(_contract), counts=dict(_counts), native_reasons=dict(_reasons),
                    records=[dict(record) for record in _records])
