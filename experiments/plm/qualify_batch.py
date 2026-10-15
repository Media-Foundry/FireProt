"""Exercise the supervised resident-batch entry against retained native outputs.

Run on DiamondHill after batch_screen.py and an isolated CLI/worker deployment.
This is entry qualification, not a paired complete-invocation speedup experiment.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('/media/IntelSSD/fireprot')
    experiment = Path(sys.argv[1]).resolve()
    suffix = sys.argv[2] if len(sys.argv) > 2 else ''
    deployment = experiment / f'qualified{suffix}'
    work = experiment / f'entry-qualification{suffix}'; work.mkdir(exist_ok=False)
    cli = deployment / 'target/release/fireprot'
    report = dict(cli_sha256=digest(cli), hip_device=6,
                  policy='bitwise equality to native outputs with identical microbatch grouping',
                  measurement='entry qualification; not repeated paired E2E', runs=[])
    profiles = {'atlaslm-3b': 'plm-atlaslm-3b-optimized', 'esmc-fold2-6b': 'plm-esmc-fold2-6b-optimized',
                'esm2-3b': 'plm-esm2-3b', 'esm3-open': 'plm-esm3-open'}
    for name, profile in profiles.items():
        base = json.loads((root/f'configs/requests/plm-{name}-n1000-native.json').read_text())
        cfg = json.loads((root/f'configs/diamondhill/{profile}.json').read_text())
        cfg['workers'] = str(deployment/'python')
        cfg['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(deployment/'target/release/lib_fireprot_bridge.so')
        profile_path = work/f'{name}-profile.json'; write(profile_path, cfg)
        diagnostic = experiment/('atlaslm' if name == 'atlaslm-3b' else name)
        for case, size in [('1000-1000-1000-1000',4), ('600-800-900-1000',4), ('1000-1000-1000-1000',1)]:
            run_id = f'{name}-{case}-b{size}'
            raw_sequences = json.loads((diagnostic/case/'inputs.json').read_text())
            sequences = [dict(id=f'item-{i}', sequence=sequence) for i, sequence in enumerate(raw_sequences)]
            input_path = work/f'{run_id}-input.json'
            write(input_path, dict(schema_version=1, sequences=sequences))
            request = dict(base, run_id=run_id, output_dir=str(work/run_id), hip_device=6,
                           input=dict(path=str(input_path), sha256=digest(input_path)))
            request['workload'] = dict(base['workload'], batch=dict(items=4, max_batch_size=size,
                max_padded_tokens=4008, grouping='length_sorted'))
            request_path = work/f'{run_id}-request.json'; write(request_path, request)
            with (work/f'{run_id}.log').open('x') as log:
                start = time.perf_counter()
                subprocess.run([str(cli), 'atlaslm' if name == 'atlaslm-3b' else 'esm', str(request_path),
                                str(profile_path)], check=True, stdout=log, stderr=subprocess.STDOUT, cwd=root)
                seconds = time.perf_counter()-start
            output = Path(request['output_dir'])
            assert (output/'accepted.json').is_file()
            artifacts = []
            for i in range(4):
                expected = diagnostic/case/f'{"single" if size == 1 else "batch"}-{i}.npz'
                actual = output/f'sample-{i}.npz'
                with zipfile.ZipFile(expected) as a, zipfile.ZipFile(actual) as b:
                    assert set(a.namelist()) == set(b.namelist())
                    assert all(a.read(k) == b.read(k) for k in a.namelist()), (name,case,size,i)
                artifacts.append(dict(index=i, expected=str(expected), expected_sha256=digest(expected),
                                      actual=str(actual), actual_sha256=digest(actual)))
            result = dict(model=name, case=case, batch_size=size, outputs_exact=True,
                          observed_invocation_seconds=seconds, request_sha256=digest(request_path),
                          profile_sha256=digest(profile_path), artifacts=artifacts)
            report['runs'].append(result); write(work/'report.json', report)
            print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
