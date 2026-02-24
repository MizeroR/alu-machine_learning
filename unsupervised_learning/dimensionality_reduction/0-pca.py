#!/usr/bin/env python3
"""Performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on X to maintain var fraction of variance

    Parameters:
    X: numpy.ndarray of shape (n, d), mean-centered
    var: fraction of variance to preserve

    Returns:
    W: numpy.ndarray of shape (d, nd)
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(var, float) or
            var <= 0 or var > 1):
        return None

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance
    explained_variance = (S ** 2)
    total_variance = np.sum(explained_variance)

    # Cumulative variance ratio
    cumulative_variance = np.cumsum(explained_variance) / total_variance

    # Find number of components to maintain desired variance
    nd = np.searchsorted(cumulative_variance, var) + 1

    # Weights matrix
    W = Vt.T[:, :nd]

    return W
