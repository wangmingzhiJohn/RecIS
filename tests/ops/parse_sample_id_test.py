import unittest

import numpy as np

from recis.nn.functional.array_ops import parse_sample_id


class RarseSampleIDTest(unittest.TestCase):
    def test_parse_sample_id(self):
        data = np.array(["ID^1", "ID^3", "ID^2"])
        max_value = parse_sample_id(data, True)
        min_value = parse_sample_id(data, False)
        self.assertTrue(max_value == 3)
        self.assertTrue(min_value == 1)


if __name__ == "__main__":
    unittest.main()
