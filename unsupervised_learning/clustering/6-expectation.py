#!/usr/bin/env python3
"""Expectation step for a GMM"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in EM algorithm

    Returns:
        g, l
    or
        None, None on failure
    """

    # Validate X
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None, None

    n, d = X.shape

    # Validate pi
    if (not isinstance(pi, np.ndarray) or
            len(pi.shape) != 1):
        return None, None

    k = pi.shape[0]

    # Validate m
    if (not isinstance(m, np.ndarray) or
            m.shape != (k, d)):
        return None, None

    # Validate S
    if (not isinstance(S, np.ndarray) or
            S.shape != (k, d, d)):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Compute weighted likelihoods
    g = np.zeros((k, n))

    # ONE LOOP (allowed)
    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i] = pi[i] * P

    # Total likelihood per data point
    likelihood = np.sum(g, axis=0)

    if np.any(likelihood == 0):
        return None, None

    # Normalize to get posteriors
    g /= likelihood

    # Log likelihood
    l = np.sum(np.log(likelihood))

    return g, l
