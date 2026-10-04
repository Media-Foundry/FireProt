"""C ABI deployment must not discover compilers or reinterpret Python extensions."""
import ctypes
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from fireprot.ops._bridge import bridge
from fireprot.backends.amd import _published, _native_hip, ck_attention, lt_bmm


def test_real_published_c_library_without_build_inputs(tmp_path, monkeypatch):
    source = tmp_path / 'answer.c'
    source.write_text('int answer(void) {return 42;}\n')
    spec = dict(build=dict(schema_version=1, package='published_c_test', target='host',
        files=[dict(name=n, path=str(p)) for n,p in [
            ('compiler','/usr/bin/cc'), ('guard',Path('target/release/fireprot-compiler-guard').resolve()), ('source',source)]],
        arguments=dict(compile=['-shared','-fPIC'],include=[],link=[]),
        environment=dict(PATH='/usr/bin:/bin'),
        versions=dict(abi='c',c_abi='answer-v1',torch=str(torch.__version__),hip=str(torch.version.hip))),
        compiler='compiler',guard='guard',sources=['source'],timeout_ms=5000,lock_timeout_ms=5000)
    build = json.loads(bridge.build_hip(json.dumps(spec),tmp_path/'cache'))
    cli = str(Path('target/release/fireprot').resolve())
    result = subprocess.run([cli,'publish-kernel',build['directory'],str(tmp_path/'bundle')],check=True,capture_output=True,text=True)
    bundle = json.loads(result.stdout)
    source.unlink()
    monkeypatch.setenv('PATH','/nonexistent')
    monkeypatch.setenv('TEST_C_BUNDLE',str(tmp_path/'bundle'))
    monkeypatch.setenv('TEST_C_BUNDLE_SHA256',bundle['manifest_sha256'])
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(gcnArchName='host'))
    library, status = _published.select_c('published_c_test','TEST_C','answer-v1')
    assert library.answer() == 42 and status['mode'] == 'published'
    assert 'module_name' not in status['kernel']
    with pytest.raises(OSError,match='compatibility'):
        _published.select_c('published_c_test','TEST_C','answer-v2')
    with pytest.raises(OSError,match='compatibility'):
        _published.load('published_c_test',str(tmp_path/'bundle'),bundle['manifest_sha256'])


@pytest.mark.parametrize('module,prefix,abi',[
    (ck_attention,'FIREPROT_CK_ATTENTION','ck-attention-v1'),
    (lt_bmm,'FIREPROT_LT_BMM','lt-packed-bmm-v1'),
])
def test_explicit_bundle_precedes_build_and_keeps_ctypes_contract(module,prefix,abi,tmp_path,monkeypatch):
    monkeypatch.setenv('FIREPROT_ROCM_HOME',str(tmp_path/'absent-compiler-and-headers'))
    monkeypatch.delenv('FIREPROT_DISABLE_CK_EXTENSION',raising=False)
    monkeypatch.setattr(torch.version,'hip','fixture')
    for key in ('_entry','_entry_exact','_entry_vector','_library','_error'):
        if hasattr(module,key):monkeypatch.setattr(module,key,None)
    monkeypatch.setattr(module,'_build',{})
    def forbidden(*args,**kwargs): pytest.fail('Explicit bundle must not compile')
    monkeypatch.setattr(_native_hip,'load',forbidden)
    monkeypatch.setenv(prefix+'_BUNDLE','/declared/bundle')
    monkeypatch.delenv(prefix+'_BUNDLE_SHA256',raising=False)
    assert module.load_extension() is None
    assert 'SHA256 is required' in module.extension_status()['error']
    monkeypatch.setattr(module,'_error',None)
    library = SimpleNamespace(ck_attention=SimpleNamespace(),ck_attention_exact1536=SimpleNamespace(),ck_attention_vector_bias1000=SimpleNamespace(),lt_packed_bmm=SimpleNamespace())
    def selected(package,environment_prefix,c_abi):
        assert environment_prefix==prefix and c_abi==abi
        return library,dict(mode='published')
    monkeypatch.setattr(_published,'select_c',selected)
    assert module.load_extension() is not None
    assert module.extension_status()['build']['mode']=='published'
    if module is ck_attention:
        assert len(library.ck_attention.argtypes)==8
        assert library.ck_attention.argtypes==library.ck_attention_exact1536.argtypes==library.ck_attention_vector_bias1000.argtypes
    else:
        assert len(library.lt_packed_bmm.argtypes)==9
