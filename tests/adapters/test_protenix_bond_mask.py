"""Verify the upstream int64 construction retains all mask values and ownership."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

# Load the production method without importing optional model/data packages.
source=Path(__file__).resolve().parents[2]/"Protenix/protenix/data/core/featurizer.py"
tree=ast.parse(source.read_text())
cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="Featurizer")
method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="get_mask_features")
namespace={"np":np,"torch":torch}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),"exec"),namespace)


@pytest.mark.parametrize('bonds', [[], [[0,6,1],[6,0,1],[3,3,1],[1,2,1],[1,2,1]]])
def test_bond_mask_retains_native_values(monkeypatch, bonds):
    class Atoms:
        centre_atom_mask = np.array([1,0,0,1,0,0,1])
        plddt_m_rep_atom_mask = 1-centre_atom_mask
        distogram_rep_atom_mask = centre_atom_mask.copy()
        modified_res_mask = plddt_m_rep_atom_mask.copy()
        def __len__(self): return 7
    atoms=Atoms()
    monkeypatch.setitem(namespace,'get_ligand_polymer_bond_mask',lambda _: bonds)
    value=namespace["get_mask_features"](SimpleNamespace(cropped_atom_array=atoms))
    expected=np.zeros((7,7))
    for i,j,_ in bonds: expected[i,j]=expected[j,i]=1
    assert torch.equal(value['bond_mask'],torch.Tensor(expected).long())
    assert value['bond_mask'].dtype==torch.int64 and value['bond_mask'].is_contiguous()
    for key,attribute in [('pae_rep_atom_mask','centre_atom_mask'),('plddt_m_rep_atom_mask','plddt_m_rep_atom_mask'),('distogram_rep_atom_mask','distogram_rep_atom_mask'),('modified_res_mask','modified_res_mask')]:
        assert torch.equal(value[key],torch.from_numpy(getattr(atoms,attribute).astype(np.int64)))
    second=namespace["get_mask_features"](SimpleNamespace(cropped_atom_array=atoms))
    value['bond_mask'][0,0]=9
    assert second['bond_mask'][0,0]==0
