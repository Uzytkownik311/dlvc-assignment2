from dlvc.batches import BatchGenerator

from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
import dlvc.ops as ops

import numpy as np
import os

import unittest

'''
Make sure the following applies: 
'''

class TestPart2(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPart2, self).__init__(*args, **kwargs)
        if os.path.basename(os.getcwd()) == "test":
            self._data_dir = "../data"
        else:
            self._data_dir = "data"

    '''
    The number of training batches is 1, if the batch size is set to the number of samples in the dataset
    '''

    def test_number_batches(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, len(dataset), False)
        self.assertEqual(len(batch_set), 1)

    '''
    The number of training batches is 16, if the batch size is set to 500
    '''
    def test_number_batches_500(self):
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 500, False)
        self.assertEqual(len(batch_set), 16)

    '''
    The data and label shapes are (500, 3072) and (500,) respectively unless for the last batch
    '''

    def test_data_shapes_in_batches(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])

        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 500, False, op)
        iter_gen = iter(batch_set)
        for i in range(0, len(batch_set)-1):
            iter_result = next(iter_gen)
            self.assertEqual(iter_result.data.shape, (500, 3072))
        print('First 15 batches have correct data shape.')
        iter_result = next(iter_gen)
        print('Last data shape is' + str(iter_result.data.shape) + '.')
        self.assertEqual(iter_result.data.shape, (500, 3072))

    def test_label_shapes_in_batches(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])

        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 500, False, op)
        iter_gen = iter(batch_set)
        for i in range(0, len(batch_set)-1):
            iter_result = next(iter_gen)
            self.assertEqual(iter_result.label.shape, (500,))
        print('First 15 batches have correct label shape.')
        iter_result = next(iter_gen)
        print('Last label shape is' + str(iter_result.label.shape) + '.')
        self.assertEqual(iter_result.label.shape, (500,))


    '''
    The data type is always np.float32 and the label type is integral(one of the np.int and np.uint variants)
    '''
    def test_data_label_datatypes(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])

        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 500, False, op)
        iter_gen = iter(batch_set)
        for i in range(0, len(batch_set)):
            iter_result = next(iter_gen)
            self.assertTrue(np.issubdtype(iter_result.data.dtype, np.float32))
            self.assertTrue(np.issubdtype(iter_result.label.dtype, np.int))
            print('Dtype of label' + str(iter_result.label.dtype) + '.')

    '''
    The first sample of the first training batch returned without shuffling has label 0 and data [116. 125. 125. 91. 101. ...]
    '''
    def test_first_sample(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])

        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 500, False, op)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        print('First sample in training set has following data: ' + str(iter_result.data[0]) + '.')
        self.assertTrue(np.issubdtype(iter_result.label[0], 0))

    '''
    The first sample of the first training batch returned with shuffling must always be different
    '''

    def test_first_sample_with_shuffling(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])

        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_set = BatchGenerator(dataset, 500, True, op)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        print ('First shuffle....')
        print('First sample in training set has following data: ' + str(iter_result.data[0]) + '.')
        print('First sample in training set has following label: ' + str(iter_result.label[0]) + '.')

        batch_set = BatchGenerator(dataset, 500, True, op)
        iter_gen = iter(batch_set)
        iter_result = next(iter_gen)
        print('Second shuffle....')
        print('First sample in training set has following data: ' + str(iter_result.data[0]) + '.')
        print('First sample in training set has following label: ' + str(iter_result.label[0]) + '.')


