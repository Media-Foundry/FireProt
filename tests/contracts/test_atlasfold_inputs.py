"""Input boundaries for the AtlasFold monomer benchmark."""
import json
import pytest
from benchmarks.atlasfold.profile_inference import read_sequence


def test_single_fasta(tmp_path):
    path = tmp_path/'protein.fa'
    path.write_text('>sample\nACDE\nFGH\n')
    assert read_sequence(path) == ('sample', 'ACDEFGH')


def test_reject_multiple_fasta(tmp_path):
    path = tmp_path/'protein.fa'
    path.write_text('>one\nACDE\n>two\nFGHI\n')
    with pytest.raises(ValueError, match='Exactly one'):
        read_sequence(path)


def test_reject_multiple_chain_copies(tmp_path):
    path = tmp_path/'protein.json'
    path.write_text(json.dumps([{'name': 'sample', 'sequences': [
        {'proteinChain': {'sequence': 'ACDE', 'count': 2}}]}]))
    with pytest.raises(ValueError, match='Only one chain'):
        read_sequence(path)


def test_existing_long_cases():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]/'benchmarks/protenix/inputs'
    for length in (1000, 1536):
        paths = [p for p in root.glob(f'umem_L{length:05d}_*.json')
                 if not p.name.endswith('.metadata.json')]
        assert len(paths) == 1
        name, sequence = read_sequence(paths[0])
        assert len(sequence) == length
        assert name.startswith(f'umem_L{length:05d}')
