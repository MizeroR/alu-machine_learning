#!/usr/bin/env python3
"""
This module provides a function to add two matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list of int/float: A new matrix containing
        the element-wise sums,
        or None if the input matrices are not the same shape.
    """
    if len(mat1) != len(mat2) or any(
        len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)
    ):
        return None
    return [
        [a + b for a, b in zip(r1, r2)]
        for r1, r2 in zip(mat1, mat2)]
