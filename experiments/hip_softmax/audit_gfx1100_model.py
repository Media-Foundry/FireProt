"""Independently rehash model receipts and replay Rust array comparisons.

Rust benchmark-pairs owns scheduling, isolation and numerical policy. This
offline evidence helper additionally checks full CIF bytes and HIP route use.
It can retain partial evidence after a stopped experiment without calling it a
completed benchmark or computing a speedup from incomplete observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(record, base):
    path = Path(record['path'])
    if not path.is_absolute():
        path = base / path
    assert pin(path)['sha256'] == record['sha256'], str(path)
    return path


def audit(repo, root, reference, output):
    repo, root, reference, output = map(lambda p: Path(p).resolve(), (repo, root, reference, output))
    ref_accept = json.loads((reference / 'accepted.json').read_text())
    for item in ref_accept['metadata'] + ref_accept['outputs']:
        checked(item, reference)
    ref_bundle = json.loads((reference / 'predictions.json').read_text())
    ref_prediction = ref_bundle['predictions'][0]
    ref_npz = reference / ref_prediction['archive']
    ref_cif = reference / ref_prediction['structure']
    records = []
    for path in sorted((root / 'paired').glob('measurement-*/invocation.json')):
        invocation = json.loads(path.read_text())
        accepted = checked(invocation['accepted'], path.parent)
        directory = accepted.parent
        receipt = json.loads(accepted.read_text())
        for item in receipt['metadata'] + receipt['outputs']:
            checked(item, directory)
        bundle = json.loads((directory / 'predictions.json').read_text())
        request = json.loads((directory / 'request.json').read_text())
        prediction = bundle['predictions'][0]
        replay = subprocess.run([
            str(repo / 'target/release/fireprot'), 'compare-npz',
            str(ref_npz), str(directory / prediction['archive'])],
            text=True, capture_output=True, timeout=60)
        comparison = json.loads(replay.stdout)
        assert len(comparison) == 11
        exact = replay.returncode == 0 and all(value['bitwise_equal'] and value['passed'] for value in comparison.values())
        cif = pin(directory / prediction['structure'])
        record = dict(invocation=pin(path), accepted=pin(accepted),
            backend=request['backend'], run_id=request['run_id'],
            e2e_seconds=invocation['elapsed_seconds'], model_seconds=bundle['model_seconds'],
            request_seconds=bundle['request_seconds'],
            all_11_arrays_exact=exact, comparisons=comparison,
            npz=pin(directory / prediction['archive']), cif=cif,
            full_cif_exact=cif['sha256'] == pin(ref_cif)['sha256'])
        runtime = directory / 'operator-runtime.json'
        if runtime.exists():
            operator = json.loads(runtime.read_text())
            record.update(operator_runtime=pin(runtime),
                          launches_submitted=operator['launches_submitted'],
                          variant_calls=operator['kernel']['variant_calls'])
        records.append(record)
    candidates = [item for item in records if item['backend'] == 'hip_fused']
    evidence = dict(schema_version=1, reference=pin(reference / 'accepted.json'),
        records=records, measured_invocations=len(records),
        all_measured_arrays_exact=bool(records) and all(item['all_11_arrays_exact'] for item in records),
        all_measured_cifs_exact=bool(records) and all(item['full_cif_exact'] for item in records),
        all_candidates_executed_hip=bool(candidates) and all(item.get('launches_submitted', 0) > 0 for item in candidates),
        preparation=pin(root / 'preparation.json'),
        cli=pin(repo / 'target/release/fireprot'),
        limitations=['Environment inventory remains unqualified; this is experimental model evidence.',
                     'Clocks/caches were not fixed; no cold-cache or statistical-significance claim.'])
    for name in ('benchmark.json', 'comparison.json'):
        path = root / 'paired' / name
        if path.exists():
            evidence[name.removesuffix('.json')] = dict(pin=pin(path), contents=json.loads(path.read_text()))
    with output.open('x') as handle:
        json.dump(evidence, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(audit=pin(output), measured=len(records),
                          all_candidates_executed_hip=evidence['all_candidates_executed_hip'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'root', 'reference', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    audit(args.repo, args.root, args.reference, args.output)
