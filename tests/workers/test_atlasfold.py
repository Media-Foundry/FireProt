"""Model-specific monomer parsing; no GPU or upstream imports."""
import json
import pytest
from python.fireprot_workers.atlasfold import read_sequence


def test_fasta_and_existing_protenix_case_preserve_sequence(tmp_path):
    fasta = tmp_path / 'input.fasta'
    fasta.write_text('>protein\nACD\nEFX\n')
    assert read_sequence(fasta) == ('protein', 'ACDEFX')
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'name': 'protein', 'sequences': [
        {'proteinChain': {'sequence': 'ACDEFX', 'count': 1}}]}]))
    assert read_sequence(path) == ('protein', 'ACDEFX')


@pytest.mark.parametrize('content', ['', '>a\nACD\n>b\nEFG', '>a\nacd', '>a\nACB', '>\nACD'])
def test_unsupported_fasta_inputs_fail(tmp_path, content):
    path = tmp_path / 'input.fasta'
    path.write_text(content)
    with pytest.raises(ValueError):
        read_sequence(path)


def test_multimer_case_is_not_silently_reduced(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'name': 'protein', 'sequences': [
        {'proteinChain': {'sequence': 'ACD', 'count': 2}}]}]))
    with pytest.raises(ValueError):
        read_sequence(path)


def test_input_snapshot_keeps_consumed_bytes_when_source_changes(tmp_path):
    from python.fireprot_workers.atlasfold import snapshot_input
    source = tmp_path / 'original.json'
    original = json.dumps([{'name': 'fixture', 'sequences': [
        {'proteinChain': {'sequence': 'ACDX', 'count': 1}}]}]).encode()
    source.write_bytes(original)
    output = tmp_path / 'output'
    output.mkdir()
    snapshot = snapshot_input(source, output)
    source.unlink()
    assert snapshot.read_bytes() == original
    assert read_sequence(snapshot) == ('fixture', 'ACDX')
    source.write_bytes(original)
    with pytest.raises(FileExistsError):
        snapshot_input(source, output)


def test_fasta_ascii_separator_whitespace_matches_upstream_strip(tmp_path):
    path = tmp_path / 'input.fa'
    path.write_bytes(b'\x1f>name\x1f\rACD\x1f\nEFX\x1f')
    assert read_sequence(path) == ('name', 'ACDEFX')
