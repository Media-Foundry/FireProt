"""Pinned Torch loaders preserve disable/version gates and avoid build discovery."""
from types import SimpleNamespace
import sys
import pytest
import torch
from fireprot.backends.amd import extension, ck_extension, pair_fusions, _published


@pytest.fixture(params=[
    (extension,'load_amd_extension','extension_status','FIREPROT_AMD_OPS','FIREPROT_DISABLE_AMD_EXTENSION','fireprot_amd_ops'),
    (ck_extension,'load_ck_extension','extension_status','FIREPROT_CK_OPS','FIREPROT_DISABLE_CK_EXTENSION','fireprot_ck_ops'),
    (pair_fusions,'load','manifest','FIREPROT_PAIR_FUSIONS','FIREPROT_DISABLE_PAIR_FUSIONS_HIP','fireprot_pair_fusions'),
])
def loader(request,monkeypatch):
    module,entry,status,prefix,disable,package=request.param
    for field in ('_extension','_error','_load_error','_build','_name'):
        if hasattr(module,field):monkeypatch.setattr(module,field,None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch,'__version__','2.12.0a0+git78258b9')
    monkeypatch.setattr(torch.version,'hip','7.14.60850')
    monkeypatch.setenv(prefix+'_BUNDLE','/pinned/bundle')
    monkeypatch.setenv(prefix+'_BUNDLE_SHA256','a'*64)
    monkeypatch.delenv(disable,raising=False)
    monkeypatch.setitem(sys.modules,'fireprot.backends.amd._native_build',None)
    monkeypatch.setitem(sys.modules,'torch.utils.cpp_extension',None)
    return getattr(module,entry),getattr(module,status),prefix,disable,package


def test_pinned_loading_without_compiler_modules(loader,monkeypatch):
    load,status,prefix,disable,package=loader
    calls=[];module=SimpleNamespace(__name__='accepted_module')
    def accepted(name,path,digest):
        calls.append((name,path,digest));return module,dict(mode='published')
    monkeypatch.setattr(_published,'load',accepted)
    assert load() is module and load() is module
    assert calls==[(package,'/pinned/bundle','a'*64)]
    assert status()['build']['mode']=='published'


@pytest.mark.parametrize('failure',['missing-pin','rejected','disabled'])
def test_explicit_failure_does_not_compile(loader,failure,monkeypatch):
    load,status,prefix,disable,_=loader
    calls=[]
    def rejected(*args):calls.append(1);raise OSError('rejected bundle')
    monkeypatch.setattr(_published,'load',rejected)
    if failure=='missing-pin':monkeypatch.delenv(prefix+'_BUNDLE_SHA256')
    if failure=='disabled':monkeypatch.setenv(disable,'1')
    assert load() is None and load() is None
    message={'missing-pin':'SHA256 is required','rejected':'rejected bundle','disabled':'disabled'}[failure]
    assert message in status()['error']
    assert calls==([1] if failure=='rejected' else [])
