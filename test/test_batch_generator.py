from dlvc.batches import BatchGenerator

from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
import dlvc.ops as ops

import numpy as np
import os

import unittest



class TestBatchGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBatchGenerator, self).__init__(*args, **kwargs)
        if os.path.basename(os.getcwd()) == "test":
            self._data_dir = "../data"
        else:
            self._data_dir = "data"

    def test_create_batch(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 100, False)
        self.assertEqual(len(batch_set), 80)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        self.assertEqual(iter_result.idx[0], 9)
        iter_result = next(iter_gen)
        self.assertEqual(iter_result.idx[0], 607)

    def test_shuffle(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 100, True)
        self.assertEqual(len(batch_set), 80)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        self.assertFalse(iter_result.idx[0] == 9)
        iter_result = next(iter_gen)
        self.assertFalse(iter_result.idx[0] == 607)

    def test_data_transformation(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 100, False, op)
        self.assertEqual(len(batch_gen), 80)
        iter_gen = iter(batch_gen)
        iter_result = next(iter_gen)
        self.assertEqual(iter_result.data[0].shape, (3072,))
        self.assertTrue(np.issubdtype(iter_result.data.dtype, np.float32))

    def test_type_error_exception(self):
        self.assertRaises(TypeError, BatchGenerator, [1, 2, 3], 100, False)

    def test_batch_size_is_not_integer_exception(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TEST)
        self.assertRaises(TypeError, BatchGenerator, dataset, 50.5, False)

    def test_bigger_batch_then_dataset_exception(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TEST)
        self.assertRaises(ValueError, BatchGenerator, dataset, 5000, False)

    def test_negative_batch_size_exception(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TEST)
        self.assertRaises(ValueError, BatchGenerator, dataset, -1, False)


if __name__ == "__main__":
    unittest.main()