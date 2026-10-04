"""Compatibility import root for upstream patches and historical entry points.

Maintained modules live under python/fireprot; no framework policy belongs here.
"""
from pathlib import Path as _Path

__path__ = [str(_Path(__file__).resolve().parents[1] / "python" / "fireprot")]
