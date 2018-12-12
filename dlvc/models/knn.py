from ..model import Model
from ..batches import Batch, BatchGenerator
from ..datasets.pets import PetsDataset
from ..dataset import Subset
from .. import ops

from operator import itemgetter

import numpy as np
import os


def softmax(w: tuple, t=1.0) -> tuple:
    """Calculate the softmax of a list of numbers w.
    Parameters:
    w : list of numbers
    t : float
    Return: a list of the same length as w of non-negative numbers
    """

    e = np.exp(np.array(w) / t)
    return e / np.sum(e)


class KnnClassifier(Model):
    """
    k nearest neighbors classifier.
    Returns softmax class scores (see lecture slides).
    """

    def __init__(self, k: int, input_dim: int, num_classes: int):
        """
        Ctor.
        k is the number of nearest neighbors to consult (>= 1).
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        """

        if not k >= 1:
            raise ValueError("The number of k-nearest neighbors must be at least 1, it is: " + str(k) + ".")

        if not input_dim > 0:
            raise ValueError("The length of input vector must be greater then 0, it is: " + str(input_dim) + ".")

        if not num_classes > 1:
            raise ValueError("The number of classes to classify must be must be at least 1, it is: " + str(num_classes)
                             + ".")

        if not np.issubdtype(type(k), np.integer):
            raise TypeError("The number of k-nearest neighbors must be integer number, it is: " + str(type(k)) + ".")

        if not np.issubdtype(type(input_dim), np.integer):
            raise TypeError("The length of input vector must be integer number, it is: " + str(type(input_dim)) + ".")

        if not np.issubdtype(type(num_classes), np.integer):
            raise TypeError("The number of classes to classify must be must be integer number, it is: " +
                            str(type(num_classes)) + ".")

        self.k_n_n = k
        self.input_dim = input_dim
        self.num_classes = num_classes

        self._trained_data = None
        self._trained_labels = None

    def input_shape(self) -> tuple:
        """
        Returns the expected input shape as a tuple, which is (0, input_dim).
        """
        return 0, self.input_dim

    def output_shape(self) -> tuple:
        """
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        """
        return self.num_classes,

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Train the model on batch of data.
        As training simply entails storing the data, the model is reset each time this method is called.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns 0 as there is no training loss to compute.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        """

        self.check_correctness_of_matrix_data(data)
        self.check_correctness_of_labels(labels)

        if not data.shape[0] == labels.shape[0]:
            raise RuntimeError("The batch do not have equal number of observations and labels, "
                               "observations number: " + str(len(data)) + " labels number" + str(len(labels)) + ".")

        self._trained_data = data
        self._trained_labels = labels

        return 0.0

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        """

        self.check_correctness_of_matrix_data(data)

        knn_all_img = np.empty((data.shape[0], self.num_classes))

        for i, _dat in enumerate(data):
            distances = [(np.linalg.norm(_dat - self._trained_data[i]),
                          self._trained_labels[i]) for i in range(0, len(self._trained_data))]
            distances = sorted(distances, key=itemgetter(0))
            k_nearest_neighbors = tuple(label for _, label in distances[:self.k_n_n])
            knn_count = tuple(k_nearest_neighbors.count(i) for i in range(0, self.num_classes))
            knn_all_img[i] = softmax(knn_count)

        return knn_all_img

    def check_correctness_of_matrix_data(self, data: np.ndarray):

        if not isinstance(data, np.ndarray):
            raise TypeError("The batch size is not np.ndarray type, but: " + str(type(data)) + ".")

        if not np.issubdtype(data.dtype, np.float32):
            raise TypeError("The data has not value type np.float32, data type is: " + str(data.dtype) + ".")

        if not data.shape[1] == self.input_dim:
            raise RuntimeError("Size of inputs vectors in not equal with value specified in the constructor of"
                               "classifier: " + str(data.shape[1]) + " != " + str(self.input_dim) + ".")

    def check_correctness_of_labels(self, labels: np.ndarray):
        if not isinstance(labels, np.ndarray):
            raise TypeError("The batch size is not np.ndarray type, but: " + str(type(labels)) + ".")

        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("The labels has not value type integer, labels type is: " + str(labels.dtype) + ".")

        if not (labels < self.num_classes).all():
            raise ValueError("The labels contain unknown class.")