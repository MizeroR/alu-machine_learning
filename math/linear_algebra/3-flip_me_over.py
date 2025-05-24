#!/usr/bin/env python3
"""
This module provides a function to transpose a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
