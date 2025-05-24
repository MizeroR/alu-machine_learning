#!/usr/bin/env python3
"""
This module provides a function to concatenate two 2D matrices along a
specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of list of int/float): The first 2D matrix.
        mat2 (list of list of int/float): The second 2D matrix.
        axis (int): The axis along which to concatenate (0 for rows,
        1 for columns).

    Returns:
        list of list of int/float: A new 2D matrix after concatenation,
        or
        None if the matrices cannot be concatenated.
    """
    if axis == 0:
        # Check if columns match
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        # Check if rows match
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]
    else:
        return None
