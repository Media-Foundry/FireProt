"""Ensure import overlap preserves the supervisor start gate and model scope."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

from fireprot_workers import __main__ as entry, plm, transport


class StartupGate(unittest.TestCase):
    def invoke_ready(self, command, prepare):
        events = []
        channel = transport.Channel.__new__(transport.Channel)
        channel.emit = lambda payload: events.append(payload['event'])
        channel.socket = Mock()
        channel.socket.makefile.side_effect = lambda mode: (events.append('read_start') or io.BytesIO(command))
        hip = Mock()
        def pci(buffer, size, device):
            buffer.value = b'0000:8e:00.0'
            return 0
        hip.hipDeviceGetPCIBusId.side_effect = pci
        torch = SimpleNamespace(cuda=SimpleNamespace(init=lambda: None,
            get_device_properties=lambda device: SimpleNamespace(gcnArchName='gfx90a')))
        with patch.object(transport.Path, 'read_text', return_value='/opt/rocm/lib/libamdhip64.so\n'), \
             patch.object(transport.ctypes, 'CDLL', return_value=hip):
            callback = None if prepare is None else lambda: prepare(events)
            transport.Channel.ready(channel, torch, prepare_cpu=callback)
        events.append('authorized')
        return events

    def test_imports_overlap_but_ready_returns_only_after_start(self):
        command = b'{"schema_version":1,"command":"start"}\n'
        self.assertEqual(self.invoke_ready(command, lambda events: events.append('import')),
                         ['ready', 'import', 'read_start', 'authorized'])
        self.assertEqual(self.invoke_ready(command, None), ['ready', 'read_start', 'authorized'])

    def test_invalid_or_missing_start_remains_an_error(self):
        for command in (b'', b'{"schema_version":1,"command":"stop"}\n', b'x'*4097):
            with self.subTest(command=command[:50]), self.assertRaises((ValueError, RuntimeError)):
                self.invoke_ready(command, lambda events: events.append('import'))

    def test_import_failure_does_not_reach_authorization_read(self):
        def fail(events):
            self.assertEqual(events, ['ready'])
            raise ImportError('module unavailable')
        with self.assertRaisesRegex(ImportError, 'module unavailable'):
            self.invoke_ready(b'{"schema_version":1,"command":"start"}\n', fail)


class StartupSelection(unittest.TestCase):
    def invoke(self, mode, name='esmc-fold2-6b', commit=plm.ESM_COMMIT):
        with tempfile.TemporaryDirectory() as directory:
            request = dict(backend='fireprot', workload=dict(model='plm', model_name=name),
                           source=dict(root='/source', commit=commit))
            path = Path(directory)/'request.json'
            path.write_text(json.dumps(request))
            events = []
            channel = Mock()
            def ready(torch, prepare_cpu=None):
                events.append('ready')
                if prepare_cpu is not None:
                    prepare_cpu()
                    events.append('imported')
                events.append('start')
            channel.ready.side_effect = ready
            def predict(*args):
                self.assertEqual(events[-1], 'start')
                events.append('predict')
                return {}
            esm = ModuleType('esm')
            esm.__file__ = '/source/esm/__init__.py'
            model = ModuleType('esm.models.esmc.model')
            model.EsmcModel = Mock(side_effect=AssertionError('must not construct during import'))
            fake_torch = SimpleNamespace(__version__='test', version=SimpleNamespace(hip='test'))
            with patch.dict(os.environ, {'FIREPROT_REQUEST': str(path)}), \
                 patch.dict(sys.modules, {'torch': fake_torch, 'esm': esm, 'esm.models.esmc.model': model}), \
                 patch.object(entry, 'Channel', return_value=channel), \
                 patch.object(plm, 'predict', side_effect=predict), \
                 patch.object(entry, 'announce_outputs'):
                os.environ.pop('FIREPROT_ESMC_IMPORT_OVERLAP', None)
                if mode is not None:
                    os.environ['FIREPROT_ESMC_IMPORT_OVERLAP'] = mode
                try:
                    entry.main()
                finally:
                    channel.socket.close.assert_called_once()
                    model.EsmcModel.assert_not_called()
            return events

    def test_default_and_zero_do_not_preimport(self):
        for mode in (None, '0'):
            self.assertEqual(self.invoke(mode, name='esm2-3b'), ['ready', 'start', 'predict'])

    def test_explicit_esmc_imports_then_waits_for_start(self):
        self.assertEqual(self.invoke('1'), ['ready', 'imported', 'start', 'predict'])

    def test_wrong_model_bad_mode_and_wrong_source_rejected(self):
        for mode, name, commit in (('2', 'esmc-fold2-6b', plm.ESM_COMMIT),
                                   ('1', 'esm2-3b', plm.ESM_COMMIT),
                                   ('1', 'esmc-fold2-6b', 'wrong')):
            with self.subTest(mode=mode, name=name, commit=commit), \
                 patch.object(entry.traceback, 'print_exc'), self.assertRaises(ValueError):
                self.invoke(mode, name, commit)


if __name__ == '__main__':
    unittest.main()
