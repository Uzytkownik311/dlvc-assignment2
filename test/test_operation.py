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

    def test_image_cropping(self):
        test = np.arange(81).reshape((9, 9))
        op = ops.rcrop(3, 3, 'reflect')
        test = op(test)
        self.assertTrue(test.shape, (9, 9))
        # print(test)

    def test_image_cropping_to_big_padding(self):
        test = np.arange(81).reshape((9, 9))
        op = ops.rcrop(3, 4, 'reflect')
        self.assertRaises(ValueError, op, test)


if __name__ == "__main__":
    unittest.main()
