import dlvc.ops as ops

import numpy as np

import unittest


class TestOperation(unittest.TestCase):

    def test_ravel(self):
        test = np.arange(12).reshape(2, 3, 2)
        op = ops.vectorize()
        test = op(test)
        self.assertTrue(np.array_equal(test, np.arange(12)))

    def test_hwc2chw(self):
        test = np.arange(8).reshape(2, 2, 2)
        op = ops.hwc2chw()
        test = op(test)
        self.assertTrue(np.array_equal(test, np.array([[[0, 2], [4, 6]], [[1, 3], [5, 7]]])))

    def test_chw2hwc(self):
        test = np.array([[[0, 2], [4, 6]], [[1, 3], [5, 7]]])
        op = ops.chw2hwc()
        test = op(test)
        self.assertTrue(np.array_equal(test, np.arange(8).reshape(2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
