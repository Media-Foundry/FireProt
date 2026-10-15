"""Rehash the accepted ESM3 archive and independent official-factory comparison."""
import json,hashlib,pathlib
root=pathlib.Path('/media/IntelSSD/fireprot/outputs/experiments/plm-esm3-20261015')
output=root/'plm-esm3-open-n1000-native'
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
accepted=json.loads((output/'accepted.json').read_text())
for row in accepted['metadata']+accepted['outputs']:
 p=output/row['path'];assert p.resolve().is_relative_to(output.resolve());assert pin(p)['sha256']==row['sha256']
supervisor=json.loads((output/'supervisor.json').read_text());assert supervisor['phase']=='succeeded' and supervisor['device_verified'];assert supervisor['physical_device']['pci_address']=='0000:93:00.0'
parity=json.loads((root/'factory-parity/report.json').read_text());assert all(parity['exact'].values()) and all(parity['finite'].values())
for name in ['compare-factory.json','compare-worker.json']:
 r=json.loads((root/name).read_text());assert len(r)==3 and all(v['bitwise_equal'] and v['passed'] for v in r.values())
result=dict(schema_version=1,model='esm3-open',length=1000,hip_device=7,pci_address='0000:93:00.0',seed=101,
 accepted=pin(output/'accepted.json'),supervisor=pin(output/'supervisor.json'),factory_report=pin(root/'factory-parity/report.json'),
 comparisons={name:pin(root/name) for name in ['compare-factory.json','compare-worker.json']},
 archives={name:pin(root/'factory-parity'/name) for name in ['official.npz','adapter.npz']},
 weights=parity['checkpoint'],source=parity['source'],output_descriptors=parity['arrays'],all_arrays_exact=True,
 scope='One N1000 Rust-supervised standalone forward plus independent official-factory parity: all tokens, BF16 prenorm embeddings and BF16 sequence logits exact. No generation, auxiliary codec, or acceleration claim. The forward computes the native model heads; only sequence logits/embeddings are exported.')
(root/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print('Verified accepted archive and all three native/factory/adapter outputs')
