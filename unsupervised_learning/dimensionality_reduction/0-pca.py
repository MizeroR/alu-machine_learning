#!/usr/bin/env python3
"""Performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on X to maintain var fraction of variance

    Parameters:
    X: numpy.ndarray of shape (n, d), mean-centered
    var: fraction of variance to preserve OR int for number of components

    Returns:
    W: numpy.ndarray of shape (d, nd)
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Check if var is int (number of components) or float (variance fraction)
    if isinstance(var, int):
        if var <= 0:
            return None
        n, d = X.shape
        nd = min(var, d)

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        W = Vt.T[:, :nd]
        return W
    if not isinstance(var, float) or var <= 0 or var > 1:
        return None

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance ratio
    explained_variance_ratio = (S ** 2) / np.sum(S ** 2)

    # Cumulative variance ratio
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find the minimum number of components to reach variance threshold
    # We need cumulative_variance[i] >= var for the first time
    nd = np.argmax(cumulative_variance >= var) + 1

    # Weights matrix (principal components)
    W = Vt.T[:, :nd]

    return W
