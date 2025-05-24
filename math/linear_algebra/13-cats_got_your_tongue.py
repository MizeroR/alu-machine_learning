#!/usr/bin/env python3
"""
This module provides a function to concatenate two numpy arrays
along a specified axis.
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy arrays along the specified axis.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray): The second array.
        axis (int): The axis along which to concatenate.

    Returns:
        numpy.ndarray: The concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)
