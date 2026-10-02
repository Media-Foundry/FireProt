from __future__ import annotations

import os
from unittest import mock

from fireprot.adapters.protenix import hooks


def test_ck_ops_require_explicit_opt_in() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FIREPROT_ENABLE_CK_OPS", None)
        os.environ.pop("FIREPROT_DISABLE_CK_OPS", None)
        assert hooks._ck_ops_enabled() is False


def test_ck_ops_disable_override_wins() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "FIREPROT_ENABLE_CK_OPS": "1",
            "FIREPROT_DISABLE_CK_OPS": "1",
        },
    ):
        assert hooks._ck_ops_enabled() is False


def test_ck_ops_can_be_enabled_for_candidate_measurement() -> None:
    with mock.patch.dict(os.environ, {"FIREPROT_ENABLE_CK_OPS": "1"}, clear=False):
        os.environ.pop("FIREPROT_DISABLE_CK_OPS", None)
        assert hooks._ck_ops_enabled() is True
