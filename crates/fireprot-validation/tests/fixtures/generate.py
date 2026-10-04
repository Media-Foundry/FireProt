"""Regenerate small NumPy interoperability fixtures; not runtime framework code."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).parent
values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
np.save(root / 'fortran_be.npy', np.asfortranarray(values.astype('>f4')))
coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
np.savez_compressed(root / 'mapping.npz', coordinates=coordinates,
                    mask=np.array([True, False, True, True]), reference=np.full_like(coordinates, -999))
rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
candidate = coordinates @ rotation + np.array([4, -2, 7])
candidate[3] += np.array([0.001, -0.002, 0.003])
mobile = candidate - candidate.mean(0)
target = coordinates - coordinates.mean(0)
u, _, vt = np.linalg.svd(mobile.T @ target)
if np.linalg.det(u @ vt) < 0:
    u[:, -1] *= -1
metrics = dict(raw_rmsd_angstrom=float(np.sqrt(np.mean(np.sum((candidate-coordinates)**2,axis=-1)))),
               aligned_rmsd_angstrom=float(np.sqrt(np.mean(np.sum((mobile @ (u @ vt)-target)**2,axis=-1)))),
               max_abs_coordinate_delta_angstrom=float(np.abs(candidate-coordinates).max()))
(root / 'alignment.json').write_text(json.dumps(dict(reference=coordinates.tolist(), candidate=candidate.tolist(), metrics=metrics), indent=2)+'\n')
# ZIP readers may hide duplicate member names; the Rust reader must reject them.
import io
import warnings
import zipfile
stream = io.BytesIO()
np.save(stream, np.array([1.], dtype=np.float32))
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    with zipfile.ZipFile(root / 'duplicate.npz', 'w') as archive:
        archive.writestr('coordinates.npy', stream.getvalue())
        archive.writestr('coordinates.npy', stream.getvalue())
arrays = {'coordinates': np.zeros((4, 3), dtype=np.float32),
          'confidence.plddt': np.ones(4, dtype=np.float32),
          'confidence.pde': np.ones((2, 2), dtype=np.float32),
          'confidence.pae': np.ones((2, 2), dtype=np.float32)}
for key in ['ptm', 'iptm', 'gpde', 'disorder', 'has_clash', 'sample_ranking_score', 'chain_ptm.1']:
    arrays['confidence.' + key] = np.asarray(0.5, dtype=np.float32)
np.savez(root / 'openfold_sample.npz', **arrays)
