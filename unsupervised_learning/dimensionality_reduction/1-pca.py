#!/usr/bin/env python3
"""PCA v2 function that reduces to a fixed number of dimensions"""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset.

    X: numpy.ndarray of shape (n, d)
    ndim: new dimensionality
    Returns: T, numpy.ndarray of shape (n, ndim)
    """
    X_m = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_m)
    W = Vt[:ndim].T
    T = np.matmul(X_m, W)
    return T
