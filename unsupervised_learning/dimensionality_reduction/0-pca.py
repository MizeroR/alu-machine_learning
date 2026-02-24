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

    # Compute explained variance ratio
    explained_variance_ratio = (S ** 2) / np.sum(S ** 2)

    # Cumulative variance ratio
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find number of components needed to reach var
    # searchsorted finds where var would be inserted to maintain order  
    # This gives us the index of the first component where cumsum >= var
    nd = np.searchsorted(cumulative_variance, var) + 1

    # Handle edge case where we need all components
    nd = min(nd, len(S))

    # Weights matrix (principal components)
    W = Vt.T[:, :nd]

    return W
