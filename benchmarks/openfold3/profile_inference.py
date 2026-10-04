"""Pinned OpenFold3/OpenBind-0 native request and inclusive GPU stage probe."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ASSETS = Path('/media/PM982/engramfold/folding_e2e_20260921')
SOURCE_COMMIT = '7de748b7bc93adb5af0a4032d5e9f208b6e1325f'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('native.yml'))
    p.add_argument('--stages', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import numpy as np
    import openfold3
    from benchmarks.atlasfold.profile_inference import resolve_physical_device, StageEvents, _GpuProcessMonitor
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from openfold3.run_openfold import cli
    source = Path(openfold3.__file__).resolve().parent.parent
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == SOURCE_COMMIT
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    weight = ASSETS/'weights/openbind0/of3-ob-2025-06-30-174k.pt'
    report = {'status':'running','source_commit':SOURCE_COMMIT,'source':str(source),
              'weight':{'path':str(weight),'sha256':sha256(weight),'bytes':weight.stat().st_size},
              'input_sha256':sha256(args.input),'config_yaml':args.config.read_text(),
              'torch':str(torch.__version__),'hip':torch.version.hip,'stage_events':args.stages,
              'harness_sha256':sha256(__file__), 'model_calls':0,
              'cache':'Weights hashed before request; filesystem warm; fresh model; no MSA or template search',
              'timing_scope':'Request excludes imports, weight hashing and device preflight; includes upstream setup/loading/features/prediction/writing and raw output capture. Stage times are inclusive.'}
    torch.cuda.init()
    device = resolve_physical_device();report['device']=device
    monitor = _GpuProcessMonitor(device['gpu'],os.getpid(),1);monitor.start()
    stages = StageEvents(torch)
    original_forward = OpenFold3.forward
    original_predict = OpenFold3AllAtom.predict_step
    def forward(model, *a, **kw):
        if report['model_calls'] == 0:
            report['model_config'] = model.config.to_dict()
            if args.stages:
                for name in ('run_trunk','_rollout'):
                    stages.wrap(model,name,name)
                for name in ('input_embedder','msa_module','pairformer_stack','sample_diffusion','aux_heads'):
                    stages.wrap(getattr(model,name),'forward',name)
        report['model_calls'] += 1
        torch.cuda.synchronize();start=time.perf_counter()
        result=original_forward(model,*a,**kw)
        torch.cuda.synchronize();report['model_seconds']=time.perf_counter()-start
        report['recycles']=result[1]['recycles']
        return result
    def predict(runner,*a,**kw):
        result=original_predict(runner,*a,**kw)
        if result is None:
            raise RuntimeError('Upstream prediction returned no output; inspect prediction error log')
        _, outputs=result
        arrays={}
        def collect(prefix,obj):
            if isinstance(obj,dict):
                for k,v in obj.items():collect(prefix+'.'+str(k),v)
            elif isinstance(obj,torch.Tensor):
                value=obj.detach().cpu()
                arrays[prefix]=value.float().numpy() if value.dtype==torch.bfloat16 else value.numpy()
            elif isinstance(obj,(float,int)):
                arrays[prefix]=np.asarray(obj)
        collect('coordinates',outputs['atom_positions_predicted'])
        collect('confidence',outputs['confidence_scores'])
        np.savez(args.out/'prediction.npz',**arrays)
        report['output_fields']={k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all())} for k,v in arrays.items()}
        return result
    OpenFold3.forward=forward;OpenFold3AllAtom.predict_step=predict
    start=time.perf_counter()
    try:
        cli.main(args=['predict','--query-json',str(args.input.resolve()),'--use-msa-server=False',
                      '--runner-yaml',str(args.config.resolve()),'--inference-ckpt-path',str(weight),
                      '--num-diffusion-samples','1','--output-dir',str((args.out/'native').resolve())],standalone_mode=False)
        torch.cuda.synchronize()
        report['request_seconds']=time.perf_counter()-start
        assert report['model_calls']==1 and (args.out/'prediction.npz').exists()
        assert all(v['finite'] for v in report['output_fields'].values())
        report['status']='complete'
    except BaseException as e:
        report['status']='failed';report['error']=repr(e);report['traceback']=traceback.format_exc();raise
    finally:
        report['stages']=stages.summarize()
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['blas_preference']=str(torch.backends.cuda.preferred_blas_library())
        report['isolation']=monitor.stop()
        OpenFold3.forward=original_forward;OpenFold3AllAtom.predict_step=original_predict
        for obj, name, original in reversed(stages.originals):
            setattr(obj, name, original)
        (args.out/'manifest.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


if __name__ == '__main__':
    main()
