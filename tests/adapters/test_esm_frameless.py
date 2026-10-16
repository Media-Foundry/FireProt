"""Guard the ESM3 no-frame shortcut; valid frames and training must stay native."""
import unittest
from types import SimpleNamespace
import torch
from fireprot.adapters.plm.esm import frameless_output


class FramelessTest(unittest.TestCase):
    def test_zero_mask_keeps_projection_bias_and_shape(self):
        projection = torch.nn.Linear(3, 4, bias=True)
        module = SimpleNamespace(mask_and_zero_frameless=True,training=False,out_proj=projection)
        s = torch.randn(2,5,4)
        mask = torch.zeros(1,5,dtype=torch.bool)
        with torch.inference_mode():
            actual,used = frameless_output(module,lambda *args:self.fail('Unexpected native work'),
                                          s,None,mask,None,None)
        self.assertTrue(used)
        self.assertEqual(tuple(actual.shape),(2,5,4))
        self.assertTrue(torch.equal(actual,projection.bias.expand_as(actual)))

    def test_valid_frames_training_and_gradients_use_native(self):
        module = SimpleNamespace(mask_and_zero_frameless=True,training=False,out_proj=None)
        s = torch.randn(1,5,4)
        marker = object()
        seen = []
        def native(*args):
            seen.append(args)
            return marker
        mask = torch.zeros(1,5,dtype=torch.bool)
        for variant in ('valid_frame','training','unmasked','gradient'):
            module.training = variant == 'training'
            module.mask_and_zero_frameless = variant != 'unmasked'
            mask.zero_()
            if variant == 'valid_frame': mask[0,2] = True
            with torch.set_grad_enabled(variant == 'gradient'):
                actual,used = frameless_output(module,native,s,None,mask,None,None)
            self.assertIs(actual,marker)
            self.assertFalse(used)
        self.assertEqual(len(seen),4)


if __name__ == '__main__': unittest.main()
