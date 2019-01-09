import numpy as np

from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''
        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''
        self._number_correct = 0
        self._number_total = 0


    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if not (prediction <= 1).all() and (prediction >= 0).all():
            raise ValueError("Prediction values must be between 0 and 1.")

        if not (prediction.shape[0] == target.shape[0]):
            raise ValueError("Prediction must have same number of values as target.")

        self._number_total += prediction.shape[0]
        self._number_correct += np.sum(np.argmax(prediction, 1) == target)


    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        return 'accuracy: ' + str(self.accuracy())

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not np.issubdtype(type(self.accuracy()), np.floating) and np.issubdtype(type(other), np.floating):
            raise TypeError("Accuracy values have different type. Left side type: " +
                            str(type(self.accuracy())) + ", right side: " + str(type(self.accuracy())) + ".")

        if self.accuracy() < other:
            return True
        return False

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not np.issubdtype(type(self.accuracy()), np.floating) and np.issubdtype(type(other), np.floating):
            raise TypeError("Accuracy values have different type. Left side type: " +
                            str(type(self.accuracy())) + ", right side: " + str(type(self.accuracy())) + ".")

        if self.accuracy() > other:
            return True
        return False

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        if self._number_total == 0:
            return 0
        else:
            return self._number_correct/self._number_total

