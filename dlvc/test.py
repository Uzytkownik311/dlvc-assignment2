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
        self.prediction = None
        self.target = None
        self.accuracy_value = 0.
        self.true_predictions = 0

    def reset(self):
        '''
        Resets the internal state.
        '''

        self.prediction = None
        self.target = None
        self.accuracy_value = 0.
        self.true_predictions = 0

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

        self.prediction = prediction
        self.target = target

        outcome = np.argmax(self.prediction, axis=1)
        for oc, tr in zip(outcome, self.target):
            if oc == tr:
                self.true_predictions += 1

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        return 'accuracy: ' + str(self.accuracy_value)

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not np.issubdtype(type(self.accuracy_value), np.floating) and np.issubdtype(type(other), np.floating):
            raise TypeError("Accuracy values have different type. Left side type: " +
                            str(type(self.accuracy_value)) + ", right side: " + str(type(self.accuracy_value)) + ".")

        if self.accuracy_value < other:
            return True
        return False

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not np.issubdtype(type(self.accuracy_value), np.floating) and np.issubdtype(type(other), np.floating):
            raise TypeError("Accuracy values have different type. Left side type: " +
                            str(type(self.accuracy_value)) + ", right side: " + str(type(self.accuracy_value)) + ".")

        if self.accuracy_value > other:
            return True
        return False

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        self.accuracy_value = self.true_predictions/len(self.prediction)
        return self.accuracy_value
