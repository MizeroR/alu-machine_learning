#!/usr/bin/env python3
"""
This module provides a function that performs matrix multiplication
between two 2D matrices.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices.

    Args:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list of int/float: The result of the matrix
        multiplication,
        or None if the matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            s = 0
            for k in range(len(mat2)):
                s += mat1[i][k] * mat2[k][j]
            row.append(s)
        result.append(row)

    return result
