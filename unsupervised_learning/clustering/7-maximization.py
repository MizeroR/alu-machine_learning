#!/usr/bin/env python3
"""Maximization step for a GMM"""

import numpy as np


def maximization(X, g):
    """
    Performs the maximization step in EM algorithm

    Returns:
        pi, m, S
    or
        None, None, None on failure
    """

    # Validate X
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None, None, None

    n, d = X.shape

    # Validate g
    if (not isinstance(g, np.ndarray) or
            len(g.shape) != 2):
        return None, None, None

    k, n_g = g.shape

    if n_g != n:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    # Effective number of points per cluster
    Ni = np.sum(g, axis=1)  # (k,)

    if np.any(Ni == 0):
        return None, None, None

    # Update priors
    pi = Ni / n  # (k,)

    # Update means (fully vectorized)
    m = (g @ X) / Ni[:, np.newaxis]  # (k, d)

    # Update covariance matrices
    S = np.zeros((k, d, d))

    # ONE LOOP (allowed)
    for i in range(k):
        diff = X - m[i]  # (n, d)
        weighted = g[i][:, np.newaxis] * diff
        S[i] = (weighted.T @ diff) / Ni[i]

    return pi, m, S
