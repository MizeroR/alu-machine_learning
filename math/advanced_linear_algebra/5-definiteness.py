#!/usr/bin/env python3
"""
This module provides a function to determine the definiteness
of a symmetric matrix using its eigenvalues.
"""


import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a real symmetric matrix.

    Args:
        matrix (numpy.ndarray): A 2D NumPy array.

    Returns:
        str or None: One of the following strings if the matrix is symmetric:
            - "Positive definite"
            - "Positive semi-definite"
            - "Negative definite"
            - "Negative semi-definite"
            - "Indefinite"
        Otherwise, returns None.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    # Check if matrix is square and 2D
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Compute eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0)
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0)

    if pos:
        return "Positive definite"
    if pos_semi:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if neg_semi:
        return "Negative semi-definite"

    return "Indefinite"
