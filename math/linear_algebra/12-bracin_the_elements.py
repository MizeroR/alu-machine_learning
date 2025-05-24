#!/usr/bin/env python3
"""
This module provides a function to perform element-wise addition,
subtraction, multiplication, and division of numpy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    on two numpy arrays.

    Args:
        mat1: numpy.ndarray or compatible
        mat2: numpy.ndarray or compatible

    Returns:
        tuple: (sum, difference, product, quotient) of element-wise operations.
    """
    return (mat1 + mat2,
            mat1 - mat2,
            mat1 * mat2,
            mat1 / mat2)
