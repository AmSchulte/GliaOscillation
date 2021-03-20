import numpy as np


def detrend(array: np.array):
    """
    this function removes the slope between the beginning and end of an array.
    we do this by applying basic math, there is probably a function out there that could do that for us.
    anyway, its basic and i am typing way to much about that.

    sincerely

    phil


    array: numpy array
    detrended: numpy array
    """

    x1 = 0
    y1 = array[0]
    x2 = len(array)
    y2 = array[-1]

    # a * x + b
    # we ignore b
    # a = dy/dxF

    a = (y2 - y1) / (x2 - x1)
    fx = np.arange(len(array)) * a

    detrended = array - fx

    return detrended
