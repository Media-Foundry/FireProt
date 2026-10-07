"""CPU checks preventing an empty/corrupt trained-input screen from passing."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from screen import verified_captures


class Captures(unittest.TestCase):
    def test_missing_and_corrupt_inputs_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pins = {}
            for direction in ('start', 'end'):
                name = f'trunk_pairformer-tri_att_{direction}.pt'
                data = direction.encode()
                pins[name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
            manifest = root / 'pins.json'
            manifest.write_text(json.dumps(pins))
            with self.assertRaisesRegex(ValueError, 'Both pinned'):
                verified_captures(root, manifest)
            for name in pins:
                direction = name.removesuffix('.pt').rsplit('_', 1)[1]
                (root / name).write_bytes(direction.encode())
            paths, actual = verified_captures(root, manifest)
            self.assertEqual(len(paths), 2)
            self.assertEqual(actual, pins)
            damaged = root / 'trunk_pairformer-tri_att_start.pt'
            damaged.write_bytes(b'xxxxx')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                verified_captures(root, manifest)
            damaged.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'size differs'):
                verified_captures(root, manifest)


if __name__ == '__main__':
    unittest.main()
