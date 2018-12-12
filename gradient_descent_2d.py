
import cv2
import numpy as np

import os
import time
from collections import namedtuple
from math import floor

Vec2 = namedtuple('Vec2', ['x1', 'x2'])


class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self._fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self._fn = self._fn.astype(np.float32)
        self._fn /= (2**16-1)

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization as a color image.
        Use the result to visualize the progress of gradient descent.
        '''

        vis = self._fn - self._fn.min()
        vis /= self._fn.max()
        vis *= 255
        vis = vis.astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)

        return vis

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        # TODO implement
        # you can simply round and map to integers. if so, make sure not to set eps and step_size too low
        # for bonus points you can implement some form of interpolation (linear should be sufficient)

        self.check_localization(loc)

        shift = (loc[0] - floor(loc[0]), loc[1] - floor(loc[1]))
        base_x = int(floor(loc[0]))
        base_y = int(floor(loc[1]))
        boundary = self._fn.shape

        if not base_x+1 > boundary[0]-1 or base_y+1 > boundary[1]-1:
            top_left_corner = self._fn[base_x][base_y]
            top_right_corner = self._fn[base_x][base_y + 1]
            bottom_left_corner = self._fn[base_x + 1][base_y]
            bottom_right_corner = self._fn[base_x + 1][base_y + 1]
            local_grid_values = np.array([[top_left_corner, top_right_corner],
                                          [bottom_left_corner, bottom_right_corner]])
            interpolated_value = bilinear_interpolation(shift, local_grid_values)

        else:
            interpolated_value = self._fn[base_x][base_y]

        return interpolated_value

    def check_localization(self, loc: Vec2):
        boundary = self._fn.shape
        if loc[0] < 0.0 or loc[1] < 0.0 or loc[0] > boundary[0]-1 or loc[1] > boundary[1]-1:
            raise ValueError("Location is beyond domain function, location: " + str(loc) +
                             " horizontal function domain: (0, " + str(boundary[0]-1) +
                             "), vertical function domain: (0, " + str(boundary[1]-1) + ").")


def grad(fn: Fn, loc: Vec2, eps: float) -> Vec2:
    '''
    Compute the numerical gradient of a 2D function fn at location loc,
    using the given epsilon. See lecture 5 slides.
    Raises ValueError if loc is out of bounds of fn or if eps <= 0.
    '''

    # TODO implement one of the two versions presented in the lecture

    fn.check_localization(loc)
    if eps <= 0:
        raise ValueError("Epsilon has negative value or equal zero.")

    if loc[0] - eps >= 0 and loc[1] - eps >= 0:
        pos_eps_loc = (loc[0], loc[1] + eps)
        neg_eps_loc = (loc[0], loc[1] - eps)
        dx = (fn(pos_eps_loc) - fn(neg_eps_loc))/2*eps
        pos_eps_loc = (loc[0] + eps, loc[1])
        neg_eps_loc = (loc[0] - eps, loc[1])
        dy = (fn(pos_eps_loc) - fn(neg_eps_loc))/2*eps

    else:
        pos_eps_loc = (loc[0], loc[1] + eps)
        dx = (fn(pos_eps_loc) - fn(loc)) / eps
        pos_eps_loc = (loc[0] + eps, loc[1])
        dy = (fn(pos_eps_loc) - fn(loc)) / eps

    return dx, dy


def bilinear_interpolation(shift: tuple, l_grid_value: np.ndarray):

    """
    It is function which applies Bilinear interpolation for one point
    (for more info: https://en.wikipedia.org/wiki/Bilinear_interpolation).
    :param shift: represents shit from pixel with the lowest value of coordinates x and y (top left corner)
    :param l_grid_value: represents values of function in considered neighbourhood (four local grid points, matrix 2x2)
    :return: interpolate value
    """

    first_vector = np.array([1.0 - shift[0], shift[0]])
    second_vector = np.array([[1.0 - shift[1]], [shift[1]]])
    first_mul = np.matmul(l_grid_value, second_vector)
    result = np.matmul(first_vector, first_mul)

    return result[0]


if __name__ == '__main__':
    # parse args

    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--step_size', type=float, default=10.0, help='Step size')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    parser.add_argument('--stop_condition', type=float, default=0.001,
                        help='Program stops if location shift will be lower then this argument value')
    args = parser.parse_args()

    # init

    fn = Fn(args.fpath)
    vis = fn.visualize()
    loc = Vec2(args.sx1, args.sx2)
    eps = args.eps
    step_size = args.step_size
    beta = args.beta
    nesterov = args.nesterov
    stop_condition = args.stop_condition

    # perform gradient descent

    cv2.namedWindow('Progress', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Progress', 800, 800)

    current_loc = loc
    v = np.zeros(2)
    while True:
        # TODO implement normal gradient descent, with momentum, and with nesterov momentum depending on the arguments (see lecture 4 slides)
        # visualize each iteration by drawing on vis using e.g. cv2.line()
        # break out of loop once done
        if nesterov:
            grad_arr = np.array(grad(fn, np.add(current_loc, v), eps))
        else:
            grad_arr = np.array(grad(fn, current_loc, eps))

        v = beta*v - step_size*grad_arr
        old_loc = np.copy(current_loc)
        current_loc += v

        draw_old_loc = tuple(int(round(i)) for i in old_loc)
        draw_current_loc = tuple(int(round(i)) for i in current_loc)
        cv2.line(vis, draw_old_loc, draw_current_loc, (0, 255, 0), 3)
        cv2.imshow('Progress', vis)
        cv2.waitKey(20)  # 20 fps, tune according to your liking

        abs_v = np.absolute(v)
        if np.amax(abs_v) < stop_condition:
            break
