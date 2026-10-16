"""Direct C ABI builds use Rust and load real host code without Torch compilation."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from fireprot.ops._bridge import bridge
from fireprot.backends.amd import _native_hip, ck_attention, lt_bmm


def test_real_c_library_bridge_and_cli_cache(tmp_path):
    source=tmp_path/'answer.c';source.write_text('int answer(void) {return 42;}\n')
    request=dict(build=dict(schema_version=1,package='host_bridge',target='host',
        files=[dict(name=n,path=str(p)) for n,p in [('compiler','/usr/bin/cc'),('guard',Path('target/release/fireprot-compiler-guard').resolve()),('source',source)]],
        arguments=dict(compile=['-shared','-fPIC'],include=[],link=[]),
        environment=dict(PATH='/usr/bin:/bin'),versions=dict(recipe='fixture',abi='c')),
        compiler='compiler',guard='guard',sources=['source'],timeout_ms=5000,lock_timeout_ms=5000)
    spec=tmp_path/'recipe.json';spec.write_text(json.dumps(request))
    result=json.loads(bridge.build_hip(spec.read_text(),tmp_path/'cache'))
    library=Path(result['directory'])/next(iter(result['receipt']['artifacts']))
    assert ctypes.CDLL(str(library)).answer()==42
    command=[str(Path('target/release/fireprot').resolve()),'build-hip',str(spec),str(tmp_path/'cache')]
    cli=json.loads(subprocess.run(command,check=True,capture_output=True,text=True).stdout)
    assert cli['cache_hit'] and cli['receipt']==result['receipt']
    source.write_text('invalid native source;')
    with pytest.raises(OSError,match='Compiler log'):
        bridge.build_hip(spec.read_text(),tmp_path/'cache')


@pytest.mark.parametrize('module',[ck_attention,lt_bmm])
def test_native_loader_retains_abi_and_link_flags(module,tmp_path,monkeypatch):
    monkeypatch.setenv('FIREPROT_ROCM_HOME',str(tmp_path))
    monkeypatch.delenv('FIREPROT_DISABLE_CK_EXTENSION',raising=False)
    (tmp_path/'include/ck').mkdir(parents=True);(tmp_path/'include/ck_tile').mkdir()
    (tmp_path/'include/ck/version.h').write_text('8d1ae90eff7d022f26019ec55b2ec6a7674b3112')
    monkeypatch.setattr(torch.version,'hip','fixture')
    for key in ('_entry','_entry_exact','_entry_vector','_library','_error'):
        if hasattr(module,key):monkeypatch.setattr(module,key,None)
    monkeypatch.setattr(module,'_build',{})
    library=SimpleNamespace(ck_attention=SimpleNamespace(),ck_attention_exact1536=SimpleNamespace(),ck_attention_vector_bias1000=SimpleNamespace(),lt_packed_bmm=SimpleNamespace())
    calls=[]
    def load(*args,**kwargs):calls.append((args,kwargs));return library,{'receipt':'rust'}
    monkeypatch.setattr(_native_hip,'load',load)
    assert module.load_extension() is not None
    assert len(calls)==1 and module.extension_status()['build']=={'receipt':'rust'}
    if module is ck_attention:
        assert calls[0][0][0]=='fireprot_ck_attention'
        assert library.ck_attention.argtypes==library.ck_attention_exact1536.argtypes==library.ck_attention_vector_bias1000.argtypes
        assert len(library.ck_attention.argtypes)==8
    else:
        assert calls[0][1]['link_flags']==[f'-L{tmp_path}/lib',f'-Wl,-rpath,{tmp_path}/lib','-lhipblaslt']
        assert calls[0][1]['libraries']==[tmp_path/'lib/libhipblaslt.so']
        assert len(library.lt_packed_bmm.argtypes)==9
    monkeypatch.setattr(module,'_entry',None)
    def fail(*args,**kwargs):raise OSError('rejected native receipt')
    monkeypatch.setattr(_native_hip,'load',fail)
    assert module.load_extension() is None
    assert 'rejected native receipt' in module.extension_status()['error']


def test_native_loader_accepts_explicit_target_and_compile_flags(tmp_path, monkeypatch):
    source = tmp_path / 'kernel.hip'
    source.write_text('extern "C" int kernel() { return 0; }\n')
    root = tmp_path / 'rocm'
    (root / 'bin').mkdir(parents=True)
    (root / 'llvm/bin').mkdir(parents=True)
    (root / 'lib').mkdir(parents=True)
    (root / 'include').mkdir(parents=True)
    for path in (root / 'bin/hipcc', root / 'llvm/bin/clang++', root / 'llvm/bin/ld.lld',
                 root / 'lib/libamdhip64.so'):
        path.touch()
    monkeypatch.setattr(torch.version, 'hip', 'fixture')
    seen = {}
    def build_hip(spec, cache):
        seen.update(json.loads(spec)['build'])
        directory = tmp_path / 'artifact'
        directory.mkdir()
        library = directory / 'kernel.so'
        library.touch()
        return json.dumps(dict(directory=str(directory), receipt=dict(artifacts={'kernel.so': 'sha'})))
    monkeypatch.setattr(bridge, 'build_hip', build_hip)
    monkeypatch.setattr(_native_hip.ctypes, 'CDLL', lambda path: path)
    monkeypatch.setattr(_native_hip.shutil, 'which', lambda name: '/usr/bin/ldd')
    actual, _ = _native_hip.load('kernel', source, root, target='gfx1100',
                                  compile_flags=['-ffp-contract=off'])
    assert actual.endswith('kernel.so')
    assert seen['target'] == 'gfx1100'
    assert '--offload-arch=gfx1100' in seen['arguments']['compile']
    assert '-ffp-contract=off' in seen['arguments']['compile']
