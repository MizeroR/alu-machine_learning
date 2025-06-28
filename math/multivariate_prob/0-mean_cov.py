#!/usr/bin/env python3
"""
Calculates the mean and covariance of a dataset.
"""

import numpy as np  # type: ignore


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): shape (n, d) data set

    Returns:
        mean (numpy.ndarray): shape (1, d), mean of the data set
        cov (numpy.ndarray): shape (d, d), covariance matrix

    Raises:
        TypeError: If X is not a 2D numpy.ndarray
        ValueError: If n < 2
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Compute the mean (shape (1, d))
    mean = np.mean(X, axis=0, keepdims=True)

    # Center the data
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov  # ✅ this must be here!
