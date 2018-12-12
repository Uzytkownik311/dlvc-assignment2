from gradient_descent_2d import bilinear_interpolation as interpolate
from gradient_descent_2d import Fn, Vec2

import numpy as np
import unittest
import os


class TestGradientDescent(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestGradientDescent, self).__init__(*args, **kwargs)
        if os.path.basename(os.getcwd()) == "test":
            self._data_dir = "../fn"
        else:
            self._data_dir = "fn"

    def test_interpolation(self):
        f_values_in_l_pixels = np.array([[20, 70], [50, 10]])
        shift = (0.2, 0.75)
        result = interpolate(shift, f_values_in_l_pixels)
        self.assertEqual(result, 50.0)

    def test_interpolation_on_point(self):
        f_values_in_l_pixels = np.array([[20, 70], [50, 10]])
        shift = (1., 0.0)
        result = interpolate(shift, f_values_in_l_pixels)
        self.assertEqual(result, 50.0)

    def test_localization_out_of_function_domain_exception(self):
        fn = Fn(os.path.join(self._data_dir, "beale.png"))
        self.assertRaises(ValueError, fn, Vec2(1500, 500))

    def test_negative_localization_out_of_function_domain_exception(self):
        fn = Fn(os.path.join(self._data_dir, "beale.png"))
        self.assertRaises(ValueError, fn, Vec2(-0.25, 500))


if __name__ == "__main__":
    unittest.main()
