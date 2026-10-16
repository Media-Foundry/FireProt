"""Validate B16 fixed FFN composition against official same-group outputs."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(value):
    p = Path(value['path'])
    assert pin(p) == value
    return json.loads(p.read_text())


def archive(output):
    accepted = json.loads((output/'accepted.json').read_text())
    for value in accepted['metadata']+accepted['outputs']:
        p = output/value['path']
        assert p.resolve().is_relative_to(output.resolve()) and pin(p)['sha256'] == value['sha256']
    prediction = json.loads((output/'predictions.json').read_text())
    arrays = []
    for i, item in enumerate(prediction['predictions']):
        assert item['sample'] == i and set(item['arrays']) == {'tokens','embeddings'}
        with zipfile.ZipFile(output/item['archive']) as f:
            arrays.append({name:f.read(name) for name in f.namelist()})
    return prediction, arrays


def main():
    work = Path(sys.argv[1]).resolve()
    results = {}
    for count in (16,64):
        case = work/f'b{count}-pairs'
        reference = work/'native64'
        _, all_anchor = archive(reference)
        assert len(all_anchor)==64
        anchor = all_anchor[:count]
        for i in range(16):
            with zipfile.ZipFile(work/'screen'/f'native-{i}.npz') as f:
                assert all_anchor[i]=={name:f.read(name) for name in f.namelist()}
        reference_request = json.loads((reference/'request.json').read_text())
        assert reference_request['backend'] == 'native'
        reference_input=json.loads(Path(reference_request['input']['path']).read_text())
        benchmark_path = case/'measurements/benchmark.json'
        benchmark = json.loads(benchmark_path.read_text())
        assert benchmark['phase'] == 'succeeded' and len(benchmark['measurements']) == 6
        for key in ('execution_plan','comparison_settings'): checked(benchmark[key])
        assert pin(Path(benchmark['order']['path'])) == benchmark['order']
        comparison = checked(benchmark['comparison'])
        assert comparison['correctness_passed']
        for item in benchmark['checks']: assert checked(item)['correctness_passed']
        provenance_anchor = None
        records = []
        for job in benchmark['measurements']:
            invocation = checked(job['invocation']); checked(job['accepted'])
            assert invocation['exit_code'] == 0 and invocation['accepted'] == job['accepted']
            out = Path(job['accepted']['path']).parent
            prediction, arrays = archive(out)
            assert arrays == anchor
            request = json.loads((out/'request.json').read_text())
            if count==64:assert request['input'] == reference_request['input']
            assert request['hip_device'] == 6 and request['seed'] == 101
            assert request['workload'] == dict(model='plm',model_name='esmc-fold2-6b',precision='native',
                batch=dict(items=count,max_batch_size=16,max_padded_tokens=16032,grouping='length_sorted'))
            retained = json.loads((out/'input-snapshot.json').read_text())
            assert all(len(s['sequence'])==1000 for s in retained['sequences'])
            assert retained['sequences']==reference_input['sequences'][:count]
            assert [p['input_id'] for p in prediction['predictions']] == [s['id'] for s in retained['sequences']]
            assert prediction['batch_groups'] == [list(range(i,i+16)) for i in range(0,count,16)]
            provenance = json.loads((out/'provenance.json').read_text())
            if provenance_anchor is None: provenance_anchor = provenance
            assert provenance == provenance_anchor
            supervisor = json.loads((out/'supervisor.json').read_text())
            assert supervisor['phase']=='succeeded' and supervisor['device_verified']
            assert supervisor['physical_device']['pci_address']=='0000:8e:00.0'
            execution = json.loads((out/'execution.json').read_text())
            assert execution['environment']['FIREPROT_PREFLIGHT_HASH_THREADS']=='8'
            assert execution['environment']['FIREPROT_WORKER_CPU_AFFINITY']=='64-127,192-255'
            candidate = job['slot']['arm']=='candidate'
            assert execution['environment']['FIREPROT_ESMC_POINTWISE']=='1'
            assert execution['environment']['FIREPROT_ESMC_FIXED_GEMM']=='0'
            assert execution['environment']['FIREPROT_ESMC_B16_FFN']==('1' if candidate else '0')
            runtimes = json.loads((out/'operator-runtime.json').read_text())
            assert len(runtimes)==count//16
            plan_initialization_seconds = []
            for group_index, runtime in enumerate(runtimes):
                assert runtime['calls']==160 and runtime['rope_calls']==80 and runtime['swiglu_calls']==80
                assert runtime['batch_size']==16
                assert runtime['library']==request['resources']['hip_esmc_large_batch']
                s=runtime['pointwise']
                assert s['residual_calls']==160 and s['vector']==4 and s['vector_swiglu']==8
                assert s['library']==request['resources']['hip_esmc_large_batch']
                assert pin(Path(s['library']['path']))==s['library']
                if candidate:
                    gemm=runtime['b16_fixed_gemm']
                    assert runtime['fixed_gemm_used'] is True
                    assert gemm['calls']==80 and gemm['algorithm']==3903 and gemm['online_tuning'] is False
                    assert gemm['library']==request['resources']['hip_esmc_b16_lt']
                    assert pin(Path(gemm['library']['path']))==gemm['library']
                    assert gemm['plan_created']==(group_index==0)
                    seconds=gemm['initialization_host_seconds']
                    assert (seconds>0 if group_index==0 else seconds==0)
                    # This qualified algorithm needs no workspace; the native
                    # support check and version guards still run at creation.
                    assert gemm['workspace_bytes']==0
                    plan_initialization_seconds.append(seconds)
                else:
                    assert 'b16_fixed_gemm' not in runtime and runtime['fixed_gemm_used'] is False
            records.append(dict(slot=job['slot'],invocation=job['invocation'],accepted=job['accepted'],
                e2e_seconds=invocation['elapsed_seconds'],model_forward_seconds=prediction['model_seconds'],
                plan_initialization_seconds=plan_initialization_seconds))
        for pair in comparison['pairs']:
            assert len(pair['samples'])==count
            for sample in pair['samples']:
                assert sample['passed'] and sample['policy']=='all-arrays-exact-v1'
                assert len(sample['exact_arrays'])==2 and all(sample['exact_arrays'].values())
        results[str(count)] = dict(benchmark=pin(benchmark_path),comparison=benchmark['comparison'],
            reference_accepted=pin(reference/'accepted.json'),all_arrays_exact=True,records=records,
            performance=comparison['performance'],medians={
                field:{arm:statistics.median(r[field] for r in records if r['slot']['arm']==arm)
                       for arm in ('baseline','candidate')} for field in ('e2e_seconds','model_forward_seconds')})
    with (work/'collected.json').open('x') as f: json.dump(dict(schema_version=1,cases=results),f,indent=2);f.write('\n')
    print(json.dumps({k:dict(performance=v['performance'],medians=v['medians']) for k,v in results.items()},indent=2))


if __name__ == '__main__': main()
