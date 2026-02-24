#!/usr/bin/env python3
"""PCA function that maintains a fraction of variance"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    X: numpy.ndarray of shape (n, d), all dimensions have mean 0
    var: fraction of variance to maintain
    Returns: W, weights matrix of shape (d, nd)
    """
    U, S, Vt = np.linalg.svd(X)
    cumvar = np.cumsum(S) / np.sum(S)
    nd = int(np.sum(cumvar < var)) + 1
    W = Vt[:nd].T
    return W
