"""Check opt-in loader selection and the backwards-compatible GPU fallback."""
import os
import unittest
from unittest.mock import Mock, patch

import fireprot_workers.plm_loading as loading
import fireprot_workers.esmc_cpu_loading as cpu_loading


class EsmcLoadingSelection(unittest.TestCase):
    def test_default_and_zero_keep_existing_gpu_loader(self):
        for value in (None, '0'):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=False):
                os.environ.pop('FIREPROT_ESMC_CPU_LOADING', None)
                if value is not None:
                    os.environ['FIREPROT_ESMC_CPU_LOADING'] = value
                reference = Mock(return_value={'state': 'gpu'})
                with patch.object(loading, '_gpu_streamed_esmc_state', reference), \
                     patch.object(cpu_loading, 'reused_pinned_esmc_state') as candidate:
                    self.assertEqual(loading.streamed_esmc_state('root', {'a': 'b'}, 'convert'), {'state': 'gpu'})
                    reference.assert_called_once_with('root', {'a': 'b'}, 'convert')
                    candidate.assert_not_called()

    def test_explicit_one_selects_bounded_pinned_loader(self):
        with patch.dict(os.environ, {'FIREPROT_ESMC_CPU_LOADING': '1'}), \
             patch.object(loading, '_gpu_streamed_esmc_state') as reference, \
             patch.object(cpu_loading, 'reused_pinned_esmc_state', return_value={'state': 'cpu'}) as candidate:
            self.assertEqual(loading.streamed_esmc_state('root', {'a': 'b'}, 'convert'), {'state': 'cpu'})
            candidate.assert_called_once_with('root', {'a': 'b'}, 'convert')
            reference.assert_not_called()

    def test_invalid_selection_does_not_start_either_loader(self):
        for value in ('', 'true', '2'):
            with self.subTest(value=value), patch.dict(os.environ, {'FIREPROT_ESMC_CPU_LOADING': value}), \
                 patch.object(loading, '_gpu_streamed_esmc_state') as reference, \
                 patch.object(cpu_loading, 'reused_pinned_esmc_state') as candidate:
                with self.assertRaises(ValueError):
                    loading.streamed_esmc_state('root', {}, 'convert')
                reference.assert_not_called()
                candidate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
