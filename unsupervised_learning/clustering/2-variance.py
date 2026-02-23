#!/usr/bin/env python3
"""Module for calculating total intra-cluster variance"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset

    Parameters:
    X (numpy.ndarray): shape (n, d)
    C (numpy.ndarray): shape (k, d)

    Returns:
    float: total variance
    None: on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(C, np.ndarray) or
            len(C.shape) != 2 or
            X.shape[1] != C.shape[1]):
        return None

    # Compute squared distances
    distances = np.linalg.norm(X[:, None] - C, axis=2) ** 2

    # Minimum squared distance per point
    min_dist = np.min(distances, axis=1)

    # Total variance
    return np.sum(min_dist)
