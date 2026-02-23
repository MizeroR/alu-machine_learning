#!/usr/bin/env python3
"""Calculates the PDF of a multivariate Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    Returns:
        P or None on failure
    """

    # Validate inputs
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None

    if (not isinstance(m, np.ndarray) or
            len(m.shape) != 1):
        return None

    if (not isinstance(S, np.ndarray) or
            len(S.shape) != 2):
        return None

    n, d = X.shape

    if (m.shape[0] != d or
            S.shape != (d, d)):
        return None

    # Determinant and inverse
    det = np.linalg.det(S)
    if det <= 0:
        return None

    inv = np.linalg.inv(S)

    # Center the data
    diff = X - m  # (n, d)

    # Mahalanobis distance term (vectorized)
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)

    # Normalization constant
    norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    P = norm_const * np.exp(exponent)

    # Enforce minimum value
    P = np.maximum(P, 1e-300)

    return P
