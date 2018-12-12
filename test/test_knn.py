from dlvc.models.knn import KnnClassifier
from dlvc.batches import BatchGenerator
from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
import dlvc.ops as ops

import numpy as np
import os

import unittest

class TestKnn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestKnn, self).__init__(*args, **kwargs)
        if os.path.basename(os.getcwd()) == "test":
            self._data_dir = "../data"
        else:
            self._data_dir = "data"

    def test_creation_with_proper_data(self):
        classifier = KnnClassifier(10, 3072, 2)
        self.assertEqual(classifier.input_shape(), (0, 3072))
        self.assertEqual(classifier.output_shape(), (2,))

    def test_wrong_value_of_knn(self):
        self.assertRaises(ValueError, KnnClassifier, 0, 3072, 2)

    def test_wrong_value_of_input_dim(self):
        self.assertRaises(ValueError, KnnClassifier, 10, 0, 2)

    def test_wrong_value_of_num_classes(self):
        self.assertRaises(ValueError, KnnClassifier, 10, 3072, 0)

    def test_wrong_type_of_knn(self):
        self.assertRaises(TypeError, KnnClassifier, 10.5, 3072, 2)

    def test_wrong_type_of_input_dim(self):
        self.assertRaises(TypeError, KnnClassifier, 10, 3072.5, 2)

    def test_wrong_type_of_num_classes(self):
        self.assertRaises(TypeError, KnnClassifier, 10, 3072, 2.5)

    def test_correctness_of_data_for_train(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        one_batch_gen = BatchGenerator(dataset, 7959, False, op)
        self.assertEqual(len(one_batch_gen), 1)
        many_batch_gen = BatchGenerator(dataset, 500, False, op)
        self.assertEqual(len(many_batch_gen), 16)
        reference = [116., 125., 125., 91., 101.]
        batch_iter = iter(many_batch_gen)
        batch_iter = next(batch_iter)
        [self.assertEqual(item, reference[i]) for i, item in enumerate(batch_iter.data[0][:5])]

    def test_train_with_proper_data(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        classifier.train(iter_result.data, iter_result.label)

    def test_train_wrong_type_of_data(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        self.assertRaises(TypeError, classifier.train, [1, 2, 3], iter_result.label)

    def test_train_with_wrong_type_of_labels(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        self.assertRaises(TypeError, classifier.train, iter_result.data, [0, 1, 0])

    def test_train_with_wrong_type_of_elements_in_data(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        wrong_data = iter_result.data.astype(np.float64)
        self.assertRaises(TypeError, classifier.train, wrong_data, iter_result.label)

    def test_train_with_wrong_type_of_elements_in_labels(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        wrong_labels = iter_result.data.astype(np.float64)
        self.assertRaises(TypeError, classifier.train, iter_result.data, wrong_labels)

    def test_train_with_unknown_label(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        iter_result.label[10] = 2
        self.assertRaises(ValueError, classifier.train, iter_result.data, iter_result.label)

    def test_train_data_incompatible_with_labels(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        incompatible_labels = np.delete(iter_result.label, 111)
        self.assertRaises(RuntimeError, classifier.train, iter_result.data, incompatible_labels)

    def test_train_wrong_vector_size_in_data(self):
        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        batch_gen = BatchGenerator(dataset, 7959, False, op)
        batch_iter = iter(batch_gen)
        iter_result = next(batch_iter)
        classifier = KnnClassifier(10, 3072, 2)
        changed_data = np.delete(iter_result.data, 100, 1)
        self.assertRaises(RuntimeError, classifier.train, changed_data, iter_result.label)

    def test_predict_with_proper_data(self):

        op = ops.chain([
            ops.vectorize(),
            ops.type_cast(np.float32)
        ])
        dataset_training = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.TRAINING)
        dataset_valid = PetsDataset(os.path.join(os.getcwd(), self._data_dir), Subset.VALIDATION)

        batch_gen_t = BatchGenerator(dataset_training, 795, False, op)
        batch_gen_v = BatchGenerator(dataset_valid, 204, False, op)

        batch_iter_t = iter(batch_gen_t)
        iter_result_t = next(batch_iter_t)

        batch_iter_v = iter(batch_gen_v)
        iter_result_v = next(batch_iter_v)

        classifier = KnnClassifier(10, 3072, 2)
        classifier.train(iter_result_t.data, iter_result_t.label)
        results = classifier.predict(iter_result_v.data)
        self.assertEqual(len(results), 204)
        for result in results:
            self.assertEqual(np.sum(result), 1.0)

if __name__ == "__main__":
    unittest.main()