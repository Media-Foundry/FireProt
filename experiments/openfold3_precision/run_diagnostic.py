"""Collect one instrumented precision trial without a quality or speed verdict."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(root, dtype):
    root = Path(root).absolute()
    assert dtype in ('float32', 'bfloat16', 'float16')
    prep = read(root / 'preparation.json')
    for item in prep['files'] + [prep[k] for k in ('cli', 'source_qualification', 'reference_npz', 'reference_cif')]:
        assert pin(item['path'])['sha256'] == item['sha256'], item['path']
    cli = prep['cli']['path']
    request = read(root / f'{dtype}-request.json')
    output = Path(request['output_dir'])
    process = subprocess.run([cli, 'measure-inference', str(root / f'{dtype}-spec.json'),
                              str(root / f'{dtype}-measurement')])
    report = read(output / 'pairformer-precision.json')
    assert report['effective_precision']['pairformer_autocast'] == dtype
    result = dict(precision=dtype, instrumented=True, quality_verdict=None,
                  speedup_verdict=None, invocation_exit_code=process.returncode,
                  precision_report=report, precision_report_pin=pin(output / 'pairformer-precision.json'),
                  policy=pin(root / f'{dtype}-policy.json'), preparation=pin(root / 'preparation.json'))
    if process.returncode:
        # Only an explicitly recorded activation failure is a completed numerical
        # experiment; missing receipts, OOM and infrastructure errors need diagnosis.
        assert dtype != 'float32' and report['nonfinite'] is not None and not report['completed']
        result.update(status='nonfinite_rejected', stderr=pin(output / 'stderr.log'))
    else:
        assert report['completed'] and report['nonfinite'] is None
        accepted = read(output / 'accepted.json')
        sealed = {item['path']: item for item in accepted['outputs']}
        assert {'pairformer-precision.json', 'precision-policy.json', 'runner.json'}.issubset(sealed)
        for item in accepted['metadata'] + accepted['outputs']:
            path = output / item['path']
            assert not Path(item['path']).is_absolute() and path.resolve().is_relative_to(output.resolve())
            assert pin(path)['sha256'] == item['sha256']
        prediction = read(output / 'predictions.json')
        comparison = subprocess.run([cli, 'compare-npz', prep['reference_npz']['path'],
                                     str(output / 'sample-0.npz')], capture_output=True, text=True)
        arrays = json.loads(comparison.stdout)
        assert len(arrays) == 11
        cif = output / prediction['predictions'][0]['structure']
        if dtype == 'float32':
            assert comparison.returncode == 0 and all(v['bitwise_equal'] for v in arrays.values())
            assert cif.read_bytes() == Path(prep['reference_cif']['path']).read_bytes()
        result.update(status='finite_outputs_deviation_assessment_pending',
            exact_array_diagnostics=arrays, exact_array_comparison_exit_code=comparison.returncode,
            accepted=pin(output / 'accepted.json'), npz=pin(output / 'sample-0.npz'), cif=pin(cif),
            invocation=pin(root / f'{dtype}-measurement/invocation.json'),
            instrumented_model_seconds=prediction['model_seconds'])
    with (root / f'{dtype}-collected.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(precision=dtype, status=result['status'], quality_verdict=None)))


if __name__ == '__main__':
    run(*sys.argv[1:])
