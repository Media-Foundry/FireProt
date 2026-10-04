"""Maintained Torch loaders preserve compile options and native fallback."""
from types import SimpleNamespace
import pytest
import torch
import torch.utils.cpp_extension
from fireprot.backends.amd import extension, ck_extension, pair_fusions, _native_build


@pytest.fixture(params=[extension, ck_extension, pair_fusions])
def loader(request, monkeypatch):
    module=request.param
    for key in ('_extension','_load_error','_error','_name','_build'):
        if hasattr(module,key):monkeypatch.setattr(module,key,None)
    for key in ('FIREPROT_DISABLE_AMD_EXTENSION','FIREPROT_DISABLE_CK_EXTENSION','FIREPROT_DISABLE_PAIR_FUSIONS_HIP'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch,'__version__','2.12.0a0+git78258b9')
    monkeypatch.setattr(torch.version,'hip','7.14.60850')
    if module is extension:return module.load_amd_extension,module.extension_status,'fireprot_amd_ops'
    if module is ck_extension:return module.load_ck_extension,module.extension_status,'fireprot_ck_ops'
    return module.load,module.manifest,'fireprot_pair_fusions'


def test_rust_receipt_and_original_options(loader,monkeypatch):
    load,status,package=loader;calls=[]
    result=SimpleNamespace(__name__='verified_module');receipt={'cache_hit':True,'receipt':{'identity':'fixture'}}
    def build(name,sources,**options):
        calls.append((name,sources,options));return result,receipt
    monkeypatch.setattr(_native_build,'load',build)
    assert load() is result and load() is result
    assert len(calls)==1 and calls[0][0]==package
    options=calls[0][2]
    if package=='fireprot_ck_ops':
        assert options['cxx_flags']==['-O3','-DUSE_ROCM','-DUSE_ROCM_CK_GEMM']
        assert options['hip_flags']==[]
    if package=='fireprot_amd_ops':assert options['hip_flags']==['-O3']
    if package!='fireprot_pair_fusions':assert str(options['include_paths'][0]).endswith('/include')
    assert status()['build'] is receipt


def test_build_failure_is_native_fallback(loader,monkeypatch):
    load,status,_=loader
    def fail(*args,**kwargs):raise OSError('invalid Rust cache')
    monkeypatch.setattr(_native_build,'load',fail)
    assert load() is None and load() is None
    assert status()['error']=='OSError: invalid Rust cache'


def test_interrupt_escapes_loader(loader,monkeypatch):
    load,status,_=loader
    def fail(*args,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr(_native_build,'load',fail)
    with pytest.raises(KeyboardInterrupt):load()
    assert status()['error'] is None


def test_protenix_manifest_records_extensions_without_building(monkeypatch):
    from fireprot.adapters.protenix import runtime
    def forbidden(*args,**kwargs):raise AssertionError('manifest must not build')
    monkeypatch.setattr(extension,'load_amd_extension',forbidden)
    monkeypatch.setattr(ck_extension,'load_ck_extension',forbidden)
    amd={'loaded':True,'build':{'receipt':'amd'}}
    ck={'loaded':False,'error':None,'build':None}
    monkeypatch.setattr(extension,'extension_status',lambda:amd)
    monkeypatch.setattr(ck_extension,'extension_status',lambda:ck)
    result=runtime.manifest()
    assert result['amd_extension'] is amd and result['ck_gemm_extension'] is ck
