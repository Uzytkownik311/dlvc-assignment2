
import numpy as np
import random

from typing import List, Callable

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)

    return op


def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    return np.ravel


def hwc2chw() -> Op:

    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.transpose(a=sample, axes=(2, 0, 1))

    return op


def chw2hwc() -> Op:
    '''
    Flip a 3D array with shape CHW to HWC.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.transpose(a=sample, axes=(1, 2, 0))

    return op


def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.add(sample, val)

    return op


def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.multiply(sample, val)

    return op


def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    # TODO implement (numpy.flip will be helpful)
    return np.fliplr


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # TODO implement
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.pad.html will be helpful
    def op(sample: np.ndarray) -> np.ndarray:
        origin_size = sample.shape
        cropped_size = (origin_size[0] - sz, origin_size[1] - sz)
        crop_point = (random.randrange(cropped_size[0]), random.randrange(cropped_size[1]))

        cropped_img = sample[crop_point[0]: crop_point[0]+sz, crop_point[1]: crop_point[1]+sz]
        padding_size = ((pad, pad), (pad, pad))

        if pad > 0:
            cropped_img = np.pad(cropped_img, padding_size, pad_mode)

        if cropped_img.shape[0] > origin_size[0] or cropped_img.shape[1] > origin_size[1]:
            raise ValueError("Image ofter cropping and padding has size: " + str(cropped_img.shape) +
                             ", its size is bigger then size of origin picture: " + str(origin_size) + ".")

        return cropped_img

    return op
