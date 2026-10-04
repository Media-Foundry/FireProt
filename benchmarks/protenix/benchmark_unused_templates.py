"""Compare CPU feature preparation with exact retained-feature and RNG checks."""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,os,pathlib,pickle,random,time


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6','7'}:parser.error('Select GPU 6 or 7 visibility; computation remains on CPU')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    root=pathlib.Path(__file__).resolve().parents[2]
    os.environ.setdefault('PROTENIX_ROOT_DIR',str(root/'runtime'));os.environ['LAYERNORM_TYPE']='torch'
    import numpy as np,torch
    from ml_collections import ConfigDict
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from protenix.model.modules.pairformer import TemplateEmbedder
    from fireprot.adapters.protenix import features
    from types import SimpleNamespace
    torch.set_num_threads(1)
    assert features._verified_consumer(),features.manifest()
    assert TemplateEmbedder.forward(SimpleNamespace(n_blocks=0),{'template_aatype':None},None)==0
    assert TemplateEmbedder.forward(SimpleNamespace(n_blocks=0),{},None)==0
    assert not torch.cuda.is_initialized()
    report=dict(schema_version=1,status='running',seed=101,model='protenix_mini_default_v0.5.0',dtype='fp32',
                scope='CPU features only; no model or GPU inference',cycles_steps_samples='Not applicable',
                cache_state='One unmeasured native call per length; process and filesystem caches warm',
                threads=1,torch=torch.__version__,numpy=np.__version__,records=[],source_sha256={})
    for name in ('benchmarks/protenix/benchmark_unused_templates.py','python/fireprot/adapters/protenix/features.py',
                 'Protenix/protenix/data/inference/infer_dataloader.py','Protenix/protenix/model/modules/pairformer.py'):
        report['source_sha256'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    def seed():random.seed(101);np.random.seed(101);torch.manual_seed(101)
    def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy()))).hexdigest()
    def hash_value(h,value):
        if isinstance(value,torch.Tensor):
            assert value.device.type=='cpu';hash_value(h,value.numpy())
        elif isinstance(value,np.ndarray):
            h.update(json.dumps([str(value.dtype),list(value.shape)]).encode())
            h.update(json.dumps(value.tolist(),sort_keys=True).encode() if value.dtype.kind in 'OU' else value.tobytes())
        elif isinstance(value,dict):
            for k,v in sorted(value.items()):h.update(str(k).encode()+b'\0');hash_value(h,v)
        else:h.update(json.dumps(value,sort_keys=True).encode())
    def digest(value):
        h=hashlib.sha256();hash_value(h,value);return h.hexdigest()
    def inventory(value,prefix=''):
        if isinstance(value,torch.Tensor):yield dict(name=prefix,shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
        elif isinstance(value,dict):
            for k,v in value.items():yield from inventory(v,f'{prefix}.{k}' if prefix else k)
    try:
        for n,key in ((1000,'umem_L01000_462bb697ecdcce2f'),(1536,'umem_L01536_6b140cc19f02d9b9')):
            path=root/'benchmarks/protenix/inputs'/f'{key}.json'
            cfg=ConfigDict(dict(input_json_path=str(path),dump_dir=str(args.output.parent),use_msa=False,use_rna_msa=False,
                  use_template=False,msa_pair_as_unpair=True,esm=dict(enable=False,model_name='esm2_t36_3B_UR50D'),
                  model_name='protenix_mini_default_v0.5.0',dtype='fp32',triangle_attention='fireprot',
                  model=dict(template_embedder=dict(n_blocks=0))))
            dataset=InferenceDataset(cfg);assert len(dataset)==1
            os.environ['FIREPROT_SKIP_UNUSED_TEMPLATES']='0';seed()
            warm=dataset.process_one(copy.deepcopy(dataset.inputs[0]));del warm;gc.collect()
            references=None
            for mode in ('native_before','skip','native_after'):
                os.environ['FIREPROT_SKIP_UNUSED_TEMPLATES']='1' if mode=='skip' else '0'
                seed();sample=copy.deepcopy(dataset.inputs[0]);before=features.manifest()['counts']
                started=time.perf_counter();data,atoms,stages=dataset.process_one(sample);elapsed=time.perf_counter()-started
                after_rng=rng();all_tensors=list(inventory(data));retained=dict(data)
                retained['input_feature_dict']={k:v for k,v in data['input_feature_dict'].items() if not k.startswith('template_')}
                hashes=dict(retained_features=digest(retained),atoms=digest({**{k:atoms.get_annotation(k) for k in atoms.get_annotation_categories()},'_coordinates':atoms.coord,'_bonds':atoms.bonds.as_array()}),rng=after_rng)
                if references is None:references=hashes
                assert hashes==references,(n,mode,hashes,references)
                counts=features.manifest()['counts'];expected='skipped_templates' if mode=='skip' else 'native_templates'
                assert counts[expected]==before[expected]+1
                row=dict(n=n,mode=mode,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),wall_seconds=elapsed,
                         stages_seconds=stages,hashes=hashes,tensors=all_tensors,total_tensor_bytes=sum(t['bytes'] for t in all_tensors),
                         template_fields=sorted(k for k in data['input_feature_dict'] if k.startswith('template_')))
                report['records'].append(row);print(json.dumps({k:row[k] for k in ('n','mode','wall_seconds','stages_seconds','total_tensor_bytes')}),flush=True)
                del data,atoms,retained,sample;gc.collect()
        assert not torch.cuda.is_initialized();report['status']='ok'
    except Exception as error:report.update(status='error',error=f'{type(error).__name__}: {error}');raise
    finally:
        report['feature_preparation']=features.manifest();report['gpu_initialized']=torch.cuda.is_initialized()
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
