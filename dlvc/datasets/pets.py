from ..dataset import Sample, Subset, ClassificationDataset

import pickle
import numpy as np
import os


def unpickle(file):
    """Load byte data from file"""
    # “unpickling” is the inverse operation,
    # whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data


def check_files_exist(file_names):
    """
    Check if paths given in input exist.
    """
    for file in file_names:
        if not os.path.exists(file):
            raise ValueError("File: ", file, " does not exist")


def adjust_data(data, labels):
    """
    Process data to obtain images in BGR
    :param data: matrix with data
    :param labels: class labels
    :return: processed data
    """
    # first axis stays the same, second axis (3072 values) is split into 3
    # and the 1024 pixels are split into 32x32 pixels
    data = data.reshape((len(data), 3, 32, 32))
    for image in data:
        image[[0, 2]] = image[[2, 0]]  # RGB --> BGR
    data = np.rollaxis(data, 1, 4)  # put 3 at the end
    labels = np.array(labels)
    return data, labels


class PetsDataset(ClassificationDataset):
    """
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    """

    def __init__(self, fdir: str, subset: Subset):

        """
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        """

        self._dir = fdir
        self._data_set = []
        self._class_number = 2
        self._train_file_number = 4
        self._subset_type = subset

        self._training_files = [self._dir + "/data_batch_{}".format(i) for i in range(1, self._train_file_number+1)]
        self._validation_files = [self._dir + "/data_batch_5"]
        self._test_files = [self._dir + "/test_batch"]
        self._label_names_file = self._dir + "/batches.meta"
        self._chosen_classes = ['cat', 'dog']

        if not os.path.exists(fdir):
            raise ValueError("Directory: " + fdir + " does not exist")
        else:
            if self._subset_type == Subset.TRAINING:
                self._load_data_set(self._training_files)

            elif self._subset_type == Subset.VALIDATION:
                self._load_data_set(self._validation_files)

            elif self._subset_type == Subset.TEST:
                self._load_data_set(self._test_files)

            else:
                raise ValueError("Unknown subset: " + str(self._subset_type))

    def _load_data_set(self, file_names):
        """
        :param file_names: files paths consisting data to load
        :return: instances from indicated classes
        """
        check_files_exist(file_names)

        data_dic = unpickle(file_names[0])
        data_set = data_dic['data']
        labels = data_dic['labels']

        for file_path in file_names[1:]:
            data_dic = unpickle(file_path)
            data_set = np.vstack((data_set, data_dic['data']))
            labels += data_dic['labels']

        self._data_set = self._extract_classes(data_set, labels)

    def _extract_classes(self, data, labels):
        """
        :param data: images which should be extracted for chosen classes
        :param labels: labels which should be extracted for chosen classes
        :return: instances of chosen classes
        """

        if not os.path.exists(self._label_names_file):
            raise ValueError("File consisting metadata: " + self._label_names_file + " does not exist")

        label_names_dic = unpickle(self._label_names_file)
        label_names = label_names_dic['label_names']

        labels_type = [label_names.index(label) for label in self._chosen_classes]
        self._class_number = len(self._chosen_classes)

        extracted_instances = []
        data, labels = adjust_data(data, labels)

        for j in range(0, len(labels)):
            if labels[j] in labels_type:
                sample = Sample(idx=j, data=data[j], label=labels_type.index(labels[j]))
                extracted_instances.append(sample)

        return extracted_instances

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """

        return len(self._data_set)

    def __getitem__(self, idx: int) -> Sample:
        """
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        """

        return self._data_set[idx]

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """

        return self._class_number
