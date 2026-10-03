"""Profile CPU-only no-MSA/template/ESM feature preparation for a long input."""
from __future__ import annotations
import argparse
import cProfile
import gc
import hashlib
import json
import os
import pathlib
import pickle
import pstats
import random
import time


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--repetitions',type=int,default=2)
    args=parser.parse_args()
    if args.repetitions<1:parser.error('repetitions must be positive')
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6','7','6,7','7,6'}:
        parser.error('Limit visible devices to physical GPUs 6 and/or 7')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    root=pathlib.Path(__file__).resolve().parents[2]
    os.environ.setdefault('PROTENIX_ROOT_DIR',str(root/'runtime'))
    os.environ['LAYERNORM_TYPE']='torch'
    import numpy as np
    import torch
    from ml_collections import ConfigDict
    from protenix.data.inference.infer_dataloader import InferenceDataset
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized(), 'CPU feature probe must not initialize a GPU'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cfg=ConfigDict(dict(input_json_path=str(args.input.resolve()),dump_dir=str(args.output.parent),
                        use_msa=False,use_template=False,use_rna_msa=False,msa_pair_as_unpair=True,
                        esm=dict(enable=False,model_name='esm2_t36_3B_UR50D')))
    dataset=InferenceDataset(cfg)
    if len(dataset)!=1:raise ValueError('Use a single-case input for this probe')
    report=dict(schema_version=1,status='running',scope='CPU feature preparation only; no model or GPU inference',
                input=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                seed=101,dtype='native feature dtypes',cycles_steps_samples='Not applicable',
                cache_state='First call may initialize CCD caches; later calls reuse process caches',
                threads=1,msa=False,templates=False,esm=False,torch=torch.__version__,numpy=np.__version__,
                source_sha256={},records=[])
    for name in ('benchmarks/protenix/profile_features.py','Protenix/protenix/data/inference/infer_dataloader.py',
                 'Protenix/protenix/data/inference/json_to_feature.py','Protenix/protenix/data/core/featurizer.py'):
        report['source_sha256'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    try:
        for iteration in range(args.repetitions):
            gc.collect();random.seed(101);np.random.seed(101);torch.manual_seed(101)
            profiler=cProfile.Profile();started=time.perf_counter()
            with profiler:
                data,atoms,stages=dataset.process_one(dataset.inputs[0])
            elapsed=time.perf_counter()-started
            assert not torch.cuda.is_initialized()
            stats=pstats.Stats(profiler)
            rows=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],own_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items()]
            profile_path=args.output.with_suffix(f'.{iteration}.pstats');profiler.dump_stats(str(profile_path))
            def tensors(value,prefix=''):
                if isinstance(value,torch.Tensor):
                    assert value.device.type=='cpu'
                    yield dict(name=prefix,shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
                elif isinstance(value,dict):
                    for key,item in value.items():yield from tensors(item,f'{prefix}.{key}' if prefix else key)
            tensor_rows=list(tensors(data))
            record=dict(iteration=iteration,wall_seconds=elapsed,stages_seconds=stages,profile=str(profile_path),
                        atom_count=len(atoms),token_count=int(data['N_token'].item()),
                        feature_tensors=tensor_rows,total_tensor_bytes=sum(t['bytes'] for t in tensor_rows),
                        rng_after_sha256=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy()))).hexdigest(),
                        top_own=sorted(rows,key=lambda r:r['own_seconds'],reverse=True)[:40],
                        top_cumulative=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:50])
            report['records'].append(record)
            print(json.dumps({k:record[k] for k in ('iteration','wall_seconds','stages_seconds','atom_count','token_count','total_tensor_bytes')}),flush=True)
            del data,atoms,profiler,stats,rows,tensor_rows
        report['status']='ok'
    except Exception as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['gpu_initialized']=torch.cuda.is_initialized()
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
