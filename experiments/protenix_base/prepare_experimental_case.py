"""Extract sequence and experimental CA coordinates from a pinned local mmCIF."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import gemmi
import numpy as np


def main(source, destination):
    source=Path(source);destination=Path(destination)
    raw=source.read_bytes();cif=gzip.decompress(raw)
    block=gemmi.cif.read_string(cif.decode()).sole_block()
    sequences={row[0]:''.join(gemmi.cif.as_string(row[1]).split()) for row in block.find(['_entity_poly.entity_id','_entity_poly.pdbx_seq_one_letter_code_can'])}
    sequence=sequences['1']
    if len(sequence)!=1000 or set(sequence)-set('ACDEFGHIKLMNPQRSTVWY'):raise ValueError('Expected the complete standard-AA 1000-residue entity')
    fields=['label_entity_id','label_asym_id','pdbx_PDB_model_num','label_seq_id','label_atom_id','type_symbol','label_comp_id','label_alt_id','occupancy','Cartn_x','Cartn_y','Cartn_z','auth_seq_id','auth_asym_id']
    selected={};duplicates=0
    for ordinal,row in enumerate(block.find(['_atom_site.'+f for f in fields])):
        if tuple(row[i] for i in (0,1,2,4,5))!=('1','A','1','CA','C'):continue
        idx=int(row[3])-1;occupancy=float(row[8]);alt=row[7]
        if not 0<=idx<1000 or occupancy<=0:continue
        aa=gemmi.find_tabulated_residue(row[6]).one_letter_code.upper()
        if aa!=sequence[idx]:raise ValueError(f'Sequence/coordinate residue mismatch at {idx+1}')
        xyz=[float(row[i]) for i in (9,10,11)]
        if not np.isfinite(xyz).all():raise ValueError('Nonfinite experimental coordinates')
        priority=(occupancy,alt in ('.','?'),alt=='A',-ordinal)
        entry=dict(priority=priority,xyz=xyz,label_seq_id=idx+1,comp_id=row[6],alt_id=alt,occupancy=occupancy,auth_seq_id=row[12],auth_asym_id=row[13])
        if idx in selected:duplicates+=1
        if idx not in selected or priority>selected[idx]['priority']:selected[idx]=entry
    if set(selected)!=set(range(1000)):raise ValueError('Incomplete experimental CA coverage')
    coordinates=np.array([selected[i]['xyz'] for i in range(1000)],dtype=np.float64)
    destination.mkdir(parents=True,exist_ok=False)
    request=[dict(name='pdb_5zmw_A_L1000',sequences=[dict(proteinChain=dict(sequence=sequence,count=1))])]
    (destination/'input.json').write_text(json.dumps(request,indent=2)+'\n')
    np.savez_compressed(destination/'gt_mapping.npz',coordinates=coordinates,mask=np.ones(1000,dtype=np.bool_),label_seq_id=np.arange(1,1001,dtype=np.int32))
    metadata=dict(pdb_id='5ZMW',label_entity_id='1',label_asym_id='A',model=1,length=1000,experimental_ca_count=1000,
        coordinate_scope='Experimental CA-only coordinates in label_seq_id order; use coordinates[mask]; no input reference conformation',
        source_path=str(source),source_gzip_sha256=hashlib.sha256(raw).hexdigest(),source_mmcif_sha256=hashlib.sha256(cif).hexdigest(),
        source_url='https://www.rcsb.org/structure/5ZMW',title=gemmi.cif.as_string(block.find_value('_struct.title')),
        experimental_method=gemmi.cif.as_string(block.find_value('_exptl.method')),resolution_angstrom=float(block.find_value('_refine.ls_d_res_high')),
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),gemmi=gemmi.__version__,numpy=np.__version__,
        alternate_ca_candidates=duplicates,selection='Positive occupancy; highest occupancy, then unlabelled altloc, then A, then first source row',
        context='Development regression case, not an independent or training-excluded test set. Sequence-only predictions omit source ligands and experimental-state context.',
        artifacts={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ['input.json','gt_mapping.npz']},
        residues=[{k:v for k,v in selected[i].items() if k not in ('priority','xyz')} for i in range(1000)])
    (destination/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='residues'},indent=2))

if __name__=='__main__':main(*sys.argv[1:])
