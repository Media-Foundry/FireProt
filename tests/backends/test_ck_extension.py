from __future__ import annotations

from unittest import mock

from fireprot.backends.amd import ck_extension


def test_ck_extension_can_be_disabled_without_attempting_a_build() -> None:
    ck_extension._reset_for_testing()
    with mock.patch.dict(
        ck_extension.os.environ, {"FIREPROT_DISABLE_CK_EXTENSION": "1"}
    ):
        assert ck_extension.load_ck_extension() is None

    status = ck_extension.extension_status()
    assert status["loaded"] is False
    assert status["error"] == "disabled by FIREPROT_DISABLE_CK_EXTENSION"
    ck_extension._reset_for_testing()
