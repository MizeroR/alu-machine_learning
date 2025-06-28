#!/usr/bin/env python3
"""
Calculates the correlation matrix from a covariance matrix.
"""

import numpy as np  # type: ignore


def correlation(C):
    """
    Calculates a correlation matrix.

    Args:
        C (numpy.ndarray): shape (d, d), the covariance matrix

    Returns:
        numpy.ndarray: shape (d, d), the correlation matrix

    Raises:
        TypeError: if C is not a numpy.ndarray
        ValueError: if C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    stddev = np.sqrt(np.diag(C))
    denom = np.outer(stddev, stddev)

    correlation_matrix = C / denom
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix
