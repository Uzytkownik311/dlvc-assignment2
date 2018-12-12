from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset

import os
import cv2 as cv
import numpy as np

import unittest

class TestPets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPets, self).__init__(*args, **kwargs)
        if os.path.basename(os.getcwd()) == "test":
            self._data_dir = "../data"
        else:
            self._data_dir = "data"

    def test_correctness_of_data(self):
        training_set = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        validation_set = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.VALIDATION)
        test_set = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TEST)

        # Test number of samples in the individual data sets:
        self.assertEqual(len(training_set), 7959)
        self.assertEqual(len(validation_set), 2041)
        self.assertEqual(len(test_set), 2000)

        # #Test image shape and type
        self.assertEqual(test_set[3].data.shape, (32, 32, 3))
        self.assertEqual(test_set[3].data.dtype, 'uint8')

        #Test labels of first 10 training samples
        test_samples = []
        for i in range(0, 10):
            test_samples.append(training_set[i].label)
        self.assertEqual(test_samples, [0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

        #Make sure that color channels are in BGR order by displaying images
        #Open CV follows BGR order while Matlab follows RGB order

        my_little_sweet_dog = training_set[2].data
        channels = cv.split(my_little_sweet_dog)
        my_little_sweet_blue_dog = channels[0]
        my_little_sweet_red_dog = channels[2]

        self.assertTrue(np.sum(my_little_sweet_red_dog) > np.sum(my_little_sweet_blue_dog))

        # cv.imwrite('my_little_sweet_dog.png', my_little_sweet_dog)
        # cv.imwrite('my_little_sweet_blue_dog.png', my_little_sweet_blue_dog)
        # cv.imwrite('my_little_sweet_red_dog.png', my_little_sweet_red_dog)

if __name__ == "__main__":
    unittest.main()
