import json,sys
from pathlib import Path
from fireprot.backends.amd._native_hip import load
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
_,receipt=load('fireprot_of3_pv_order_v1',Path(__file__).with_name('pv_probe.cpp'),Path('/opt/rocm'),extra_files=[__file__],versions={'c_abi':'of3-pv-order-fp32-v1'})
(out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
