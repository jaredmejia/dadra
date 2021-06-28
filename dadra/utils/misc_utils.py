import numpy as np
import time


def format_time(t):
    """Formats time into string with minutes and seconds

    :param t: The time in seconds
    :type t: float
    :return: Formatted string with minutes and seconds
    :rtype: string
    """
    time_string = time.strftime("%M:%S", time.gmtime(t))
    time_list = time_string.split(":")
    num_min = time_list[0]
    num_sec = time_list[1]
    return f"{num_min} minutes and {num_sec} seconds"


def normalize0(X):
    """Normalizes the input array with respect to the square root of the first axis

    :param X: Input array to be normalized
    :type X: numpy.ndarray
    :return: The normalized array
    :rtype: numpy.ndarray
    """
    return X / np.sqrt(X.shape[0])


def normalize1(X):
    """Normalizes the input array with respect to the average absolute values

    :param X: Input array to be normalized
    :type X: numpy.ndarray
    :return: The normalized array
    :rtype: numpy.ndarray
    """
    return X / np.abs(X).mean()


def normalize2(X):
    """Normalizes the input array with respect to the maximum of the absolute values

    :param X: Input array to be normalized
    :type X: numpy.ndarray
    :return: The normalized array
    :rtype: numpy.ndarray
    """
    return X / np.max(np.abs(X))


def normalize(X, mu=0, std=1):
    """Normalizes the input array with mean mu and standard deviation std

    :param X: Input array to be normalized
    :type X: numpy.ndarray
    :param mu: The mean, defaults to 0
    :type mu: int, optional
    :param std: The standard deviation, defaults to 1
    :type std: int, optional
    :return: The normalized array
    :rtype: nnumpy.ndarray
    """
    return (X - mu) / std
