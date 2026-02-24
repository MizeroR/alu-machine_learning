#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on X

    Parameters:
    X: numpy.ndarray of shape (n, d)
    ndim: int, new dimensionality

    Returns:
    T: numpy.ndarray of shape (n, ndim)
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(ndim, int) or ndim <= 0 or ndim > X.shape[1]:
        return None

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Perform SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project data
    T = U[:, :ndim] * S[:ndim]

    return T
