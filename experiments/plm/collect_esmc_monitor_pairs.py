"""Independently verify monitor-only pairs and official same-batch outputs."""
import json
from pathlib import Path
import statistics
import sys

from collect_esmc_pointwise_pairs import archive, checked, pin


def main():
    work=Path(sys.argv[1]).resolve();results={}
    direct=len(sys.argv)>2 and sys.argv[2]=='direct'
    qualification=json.loads((work/'qualification-v3/qualification.json').read_text())
    assert qualification['passed']
    library=json.loads((work/'runtime-library.json').read_text());assert pin(Path(library['path']))==library
    for count in ((4,) if direct else (4,64)):
        case=work/('native-pairs' if direct else f'b{count}-pairs')
        reference=(Path('/media/IntelSSD/fireprot/outputs/experiments/esmc-load-20261016/validation/native-pairs/baseline-1') if count==4
                   else Path('/media/IntelSSD/fireprot/outputs/experiments/esmc-pointwise-20261016/validation/native64'))
        _,anchor=archive(reference);ref_request=json.loads((reference/'request.json').read_text())
        path=case/'measurements/benchmark.json';benchmark=json.loads(path.read_text())
        assert benchmark['phase']=='succeeded' and len(benchmark['measurements'])==6
        checked(benchmark['execution_plan'])
        settings=checked(benchmark['comparison_settings']);assert settings['monitor_treatment'] is True
        assert pin(Path(benchmark['order']['path']))==benchmark['order']
        comparison=checked(benchmark['comparison'])
        assert comparison['correctness_passed'] and comparison['monitor_treatment'] is True
        for item in benchmark['checks']:assert checked(item)['correctness_passed']
        records=[];provenance_anchor=None
        for job in benchmark['measurements']:
            invocation=checked(job['invocation']);checked(job['accepted'])
            assert invocation['exit_code']==0 and invocation['accepted']==job['accepted']
            out=Path(job['accepted']['path']).parent
            prediction,actual=archive(out);assert actual==anchor
            request=json.loads((out/'request.json').read_text())
            assert request['input']==ref_request['input'] and request['workload']==ref_request['workload']
            assert request['seed']==101 and request['hip_device']==6
            provenance=json.loads((out/'provenance.json').read_text())
            if provenance_anchor is None:provenance_anchor=provenance
            assert provenance==provenance_anchor
            assert prediction['batch_groups']==[list(range(i,i+4)) for i in range(0,count,4)]
            supervisor=json.loads((out/'supervisor.json').read_text())
            assert supervisor['phase']=='succeeded' and supervisor['device_verified'] and supervisor['isolation_samples']>=2
            assert supervisor['physical_device']['pci_address']=='0000:8e:00.0' and supervisor['physical_device']['smi_device']==4
            execution=json.loads((out/'execution.json').read_text())
            monitor,=[i for i in execution['inputs'] if i['name']=='amd-smi']
            expected=qualification['binaries']['native'] if job['slot']['arm']=='candidate' else qualification['official']
            assert monitor['canonical_path']==str(Path(expected['path']).resolve()) and monitor['sha256']==expected['sha256']
            assert pin(Path(expected['path']))==expected
            runtime=[] if request['backend']=='native' else json.loads((out/'operator-runtime.json').read_text())
            assert len(runtime)==(0 if request['backend']=='native' else count//4)
            for index,r in enumerate(runtime):
                assert r['calls']==160 and r['rope_calls']==80 and r['swiglu_calls']==80 and r['batch_size']==4
                if count==64:
                    assert r['library']==request['resources']['hip_esmc_pointwise']
                    assert r['pointwise']['residual_calls']==160 and r['pointwise']['vector']==4 and r['pointwise']['vector_swiglu']==8
                    g=r['fixed_gemm'];assert g['calls']==80 and g['algorithm']==4139 and g['plan_created']==(index==0)
                    assert g['online_tuning'] is False and g['library']==request['resources']['hip_esmc_lt']
                else:
                    assert r['library']==request['resources']['hip_esm'] and 'pointwise' not in r and 'fixed_gemm' not in r
            records.append(dict(slot=job['slot'],invocation=job['invocation'],accepted=job['accepted'],
                e2e_seconds=invocation['elapsed_seconds'],supervisor_seconds=supervisor['elapsed_seconds'],
                model_forward_seconds=prediction['model_seconds'],isolation_samples=supervisor['isolation_samples'],monitor=monitor))
        for pair in comparison['pairs']:
            assert len(pair['samples'])==count
            for sample in pair['samples']:
                assert sample['passed'] and sample['policy']=='all-arrays-exact-v1'
                assert len(sample['exact_arrays'])==2 and all(sample['exact_arrays'].values())
        results[str(count)]=dict(benchmark=pin(path),comparison=benchmark['comparison'],reference_accepted=pin(reference/'accepted.json'),
            all_arrays_exact=True,records=records,performance=comparison['performance'],medians={
                field:{arm:statistics.median(r[field] for r in records if r['slot']['arm']==arm) for arm in ('baseline','candidate')}
                for field in ('e2e_seconds','supervisor_seconds','model_forward_seconds','isolation_samples')})
    with (work/('collected-direct.json' if direct else 'collected.json')).open('x') as f:json.dump(dict(schema_version=1,cases=results,runtime_library=library),f,indent=2);f.write('\n')
    print(json.dumps({k:dict(performance=v['performance'],medians=v['medians']) for k,v in results.items()},indent=2))


if __name__=='__main__':main()
