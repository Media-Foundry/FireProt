"""Guard the exact ESM2 gfx1100 capability boundary."""
import unittest

from fireprot.adapters.plm.runtime import gfx1100_esm2_exact_route


class ESM2Gfx1100SelectionTest(unittest.TestCase):
    def test_exact_runtime_without_hip_library_is_selected(self):
        self.assertTrue(gfx1100_esm2_exact_route(
            'esm2-3b', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 4))

    def test_every_capability_dimension_fails_closed(self):
        cases = [
            ('esm3-open', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 4),
            ('esm2-3b', {'hip_esm': {}}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 4),
            ('esm2-3b', {}, 'gfx90a', '2.13.0+rocm10.0.0', '7.15.26333', 4),
            ('esm2-3b', {}, 'gfx1100', '2.12.0+rocm7.2', '7.15.26333', 4),
            ('esm2-3b', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.2.53211', 4),
            ('esm2-3b', {}, 'gfx1100', '2.13.0+rocm10.0.0', '7.15.26333', 1),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(gfx1100_esm2_exact_route(*case))


if __name__ == '__main__':
    unittest.main()
