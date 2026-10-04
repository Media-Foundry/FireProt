"""Probe exact int64 bond-mask construction in the CPU feature pipeline."""
from __future__ import annotations
import argparse, copy, gc, hashlib, inspect, json, os, pathlib, pickle, random, textwrap, time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6', '7'}:
        parser.error('Select authorized visibility 6 or 7; this probe never initializes the GPU')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    root = pathlib.Path(__file__).resolve().parents[2]
    os.environ.setdefault('PROTENIX_ROOT_DIR', str(root / 'runtime'))
    os.environ['LAYERNORM_TYPE'] = 'torch'
    os.environ['FIREPROT_SKIP_UNUSED_TEMPLATES'] = '1'
    import numpy as np
    import torch
    from ml_collections import ConfigDict
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.inference.infer_dataloader import InferenceDataset
    torch.set_num_threads(1)
    installed = Featurizer.get_mask_features
    native = installed
    original = textwrap.dedent(inspect.getsource(native))
    old_allocation = 'np.zeros((num_atoms, num_atoms))'
    old_conversion = 'torch.Tensor(\n        bond_mask_mat\n    ).long()'
    optimized_allocation = 'np.zeros((num_atoms, num_atoms), dtype=np.int64)'
    optimized_conversion = 'torch.from_numpy(bond_mask_mat)'
    if optimized_allocation in original:
        assert original.count(optimized_allocation) == original.count(optimized_conversion) == 1
        original = original.replace(optimized_allocation, old_allocation).replace(optimized_conversion, old_conversion)
        baseline_namespace = dict(native.__globals__)
        exec(compile(original, '<bond-mask-native-reference>', 'exec'), baseline_namespace)
        native = baseline_namespace['get_mask_features']
    assert original.count(old_allocation) == original.count(old_conversion) == 1
    replacement = original.replace(old_allocation, 'np.zeros((num_atoms, num_atoms), dtype=np.int64)').replace(old_conversion, 'torch.from_numpy(bond_mask_mat)')
    namespace = dict(native.__globals__)
    exec(compile(replacement, '<bond-mask-int64-probe>', 'exec'), namespace)
    candidate = namespace['get_mask_features']
    # Exercise the actual candidate method with nonempty, duplicate, reversed,
    # self and boundary bonds as well as the empty case; inference cases alone
    # have no ligand-polymer edges.
    class Atoms:
        centre_atom_mask = plddt_m_rep_atom_mask = distogram_rep_atom_mask = modified_res_mask = np.arange(7) % 2
        def __len__(self): return 7
    class Fixture:
        cropped_atom_array = Atoms()
    fixture_cases = [np.empty((0, 3), dtype=np.int64), np.array([[0,6,1],[6,0,1],[3,3,1],[1,2,1],[1,2,1]])]
    for bonds in fixture_cases:
        candidate.__globals__['get_ligand_polymer_bond_mask'] = lambda atoms: bonds
        actual = candidate(Fixture())
        expected = np.zeros((7,7))
        for i,j,_ in bonds: expected[i,j] = expected[j,i] = 1
        assert torch.equal(actual['bond_mask'], torch.Tensor(expected).long())
        assert actual['bond_mask'].dtype == torch.int64 and actual['bond_mask'].is_contiguous()
    candidate.__globals__['get_ligand_polymer_bond_mask'] = native.__globals__['get_ligand_polymer_bond_mask']
    def seed(): random.seed(101); np.random.seed(101); torch.manual_seed(101)
    def rng(): return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()))).hexdigest()
    def digest(value):
        h = hashlib.sha256()
        def add(v):
            if isinstance(v, torch.Tensor): add(v.numpy())
            elif isinstance(v, np.ndarray):
                h.update(json.dumps([str(v.dtype), list(v.shape)]).encode())
                if v.dtype.kind in 'OU': h.update(json.dumps(v.tolist(),sort_keys=True).encode())
                elif v.size: h.update(memoryview(np.ascontiguousarray(v)).cast('B'))
            elif isinstance(v, dict):
                for k,x in sorted(v.items()): h.update(str(k).encode()+b'\0'); add(x)
            else: h.update(json.dumps(v,sort_keys=True).encode())
        add(value); return h.hexdigest()
    report = dict(status='running',scope='CPU feature pipeline, no model/GPU inference',model='protenix_mini_default_v0.5.0',dtype='native feature dtypes; fp32 configuration',cycles_steps_samples='not applicable',seed=101,threads=1,cache_state='one unmeasured warmup per length; warm filesystem and process caches',torch=torch.__version__,numpy=np.__version__,native_method_sha256=hashlib.sha256(original.encode()).hexdigest(),candidate_source=replacement,fixture_cases=2,records=[])
    report['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('benchmarks/protenix/benchmark_bond_mask.py','Protenix/protenix/data/core/featurizer.py','Protenix/protenix/data/inference/infer_dataloader.py','python/fireprot/adapters/protenix/features.py')}
    try:
        for n,key in ((1000,'umem_L01000_462bb697ecdcce2f'),(1536,'umem_L01536_6b140cc19f02d9b9')):
            path=root/'benchmarks/protenix/inputs'/f'{key}.json'
            cfg=ConfigDict(dict(input_json_path=str(path),dump_dir=str(args.output.parent),use_msa=False,use_rna_msa=False,use_template=False,msa_pair_as_unpair=True,esm=dict(enable=False,model_name='esm2_t36_3B_UR50D'),model_name='protenix_mini_default_v0.5.0',dtype='fp32',triangle_attention='fireprot',model=dict(template_embedder=dict(n_blocks=0))))
            dataset=InferenceDataset(cfg);assert len(dataset)==1
            seed(); warm=dataset.process_one(copy.deepcopy(dataset.inputs[0]));del warm;gc.collect()
            reference=None
            for pair in range(3):
                for mode in (('native','int64') if pair%2==0 else ('int64','native')):
                    Featurizer.get_mask_features=native if mode=='native' else candidate
                    seed();sample=copy.deepcopy(dataset.inputs[0]);started=time.perf_counter()
                    data,atoms,stages=dataset.process_one(sample);elapsed=time.perf_counter()-started
                    state=rng();hashes=dict(features=digest(data),atoms=digest({**{k:atoms.get_annotation(k) for k in atoms.get_annotation_categories()},'_coordinates':atoms.coord,'_bonds':atoms.bonds.as_array()}),rng=state)
                    if reference is None:reference=hashes
                    assert hashes==reference,(n,pair,mode,hashes,reference)
                    bond=data['input_feature_dict']['bond_mask']
                    row=dict(n=n,pair=pair,mode=mode,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),wall_seconds=elapsed,stages_seconds=stages,hashes=hashes,bond_mask_shape=list(bond.shape),bond_mask_dtype=str(bond.dtype),bond_mask_bytes=bond.numel()*bond.element_size())
                    report['records'].append(row);print(json.dumps({k:row[k] for k in ('n','pair','mode','wall_seconds')}),flush=True)
                    del data,atoms,bond,sample;gc.collect()
            Featurizer.get_mask_features=native
        assert not torch.cuda.is_initialized();report['status']='ok'
    except Exception as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}');raise
    finally:
        Featurizer.get_mask_features=installed
        report['gpu_initialized']=torch.cuda.is_initialized()
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
