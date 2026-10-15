"""Prepare three matched N1000 Rust pairs after exact N600 qualification."""
from pathlib import Path
import hashlib
import json
import sys


def prepare(root):
    root = Path(root).resolve()
    qualification = json.loads((root / 'qualification-driver-status.json').read_text())
    assert qualification == dict(exit_code=0, all12_exact=True, full_cif_exact=True)
    destination = root / 'n1000-pairs'
    destination.mkdir()

    def pin(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def save(name, value):
        path = destination / name
        with path.open('x') as f:
            json.dump(value, f, indent=2)
            f.write('\n')
        return pin(path)

    profile = json.loads((root / 'qualification-profile.json').read_text())
    profile['environment']['FIREPROT_ESMFOLD2_PROFILE'] = '0'
    profile_pin = save('profile.json', profile)
    input_path = Path('/media/IntelSSD/fireprot/benchmarks/protenix/inputs/umem_L01000_462bb697ecdcce2f.json')
    assert len(json.loads(input_path.read_text())[0]['sequences'][0]['proteinChain']['sequence']) == 1000
    pairs = []
    for pair in range(3):
        arms = {}
        for arm, backend in [('baseline', 'native'), ('candidate', 'fireprot')]:
            request = json.loads((root / (backend + '-request.json')).read_text())
            tag = f'esmfold2-n1000-{backend}-pair{pair + 1}-{root.name.removeprefix("esmfold2-integration-")}'
            request.update(run_id=tag, output_dir='/media/IntelSSD/fireprot/outputs/' + tag, input=pin(input_path))
            assert not Path(request['output_dir']).exists()
            arms[arm] = dict(request=save(f'{arm}-{pair + 1}.json', request), profile=profile_pin)
        pairs.append(dict(first='candidate' if pair == 1 else 'baseline', **arms))
    bridge = pin(root / 'target/release/lib_fireprot_bridge.so')
    specification = dict(schema_version=1,
        execution=dict(schema_version=1, timeout_seconds=7200, invocation_timeout_seconds=1800, pairs=pairs),
        comparison=dict(scope='end_to_end', cache_state='Fresh model processes; filesystem and GPU library caches not flushed; N600 qualification precedes the N1000 experiment.',
            treatment_description='Pinned Biohub ESMFold2 native chunked einsum versus opt-in reuse of right operand packing, including bundled ESMC and unchanged native precision/refinement/sampling.',
            expected_changes={'backend': dict(baseline='native', candidate='fireprot'),
                'launch.inputs.rust-bridge': dict(baseline=None, candidate=dict(name='rust-bridge', canonical_path=bridge['path'], sha256=bridge['sha256']))}))
    save('specification.json', specification)
    save('preparation.json', dict(cli=pin(root / 'target/release/fireprot'), qualification=pin(root / 'qualification-result.json')))


if __name__ == '__main__':
    prepare(sys.argv[1])
