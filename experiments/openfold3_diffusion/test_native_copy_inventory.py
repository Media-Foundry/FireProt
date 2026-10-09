"""A qualified source tree must not acquire unrecorded importable code."""
import pytest

from experiments.openfold3_diffusion.prepare_native_comparison import pin, verify_copy_inventory


def test_added_or_removed_source_rejects_snapshot(tmp_path):
    package = tmp_path / "python/fireprot_workers"
    package.mkdir(parents=True)
    entry = package / "openfold3.py"
    entry.write_text("qualified")
    references = [pin(entry)]
    verify_copy_inventory(tmp_path, references)
    added = package / "extra.py"
    added.write_text("unrecorded")
    with pytest.raises(ValueError, match="inventory changed"):
        verify_copy_inventory(tmp_path, references)
    added.unlink()
    entry.unlink()
    with pytest.raises(ValueError, match="inventory changed"):
        verify_copy_inventory(tmp_path, references)
