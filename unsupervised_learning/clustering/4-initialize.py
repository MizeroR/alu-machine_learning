#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes GMM parameters

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

    # Validate k
    if (not isinstance(k, int) or
            k <= 0 or
            k > n):
        return None, None, None

    # Initialize priors evenly
    pi = np.full((k,), 1 / k)

    # Initialize means using K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Initialize covariance matrices as identity matrices
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
