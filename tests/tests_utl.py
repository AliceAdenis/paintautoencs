import os
import unittest
import tempfile

from utl import folder


class UTL(unittest.TestCase):

    def test_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, 'level_1', 'level_2', 'level_3')
            folder(test_path)
            self.assertTrue(os.path.isdir(test_path))


if __name__ == '__main__':
    unittest.main()
