#!/usr/bin/env python3
"""Module for performing K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on dataset X

    Returns:
    C, clss or (None, None) on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # First uniform call
    C = np.random.uniform(min_vals, max_vals, (k, d))

    for _ in range(iterations):  # Loop 1
        # Compute squared distances
        distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for i in range(k):  # Loop 2
            points = X[clss == i]

            if points.shape[0] == 0:
                # Second uniform call
                new_C[i] = np.random.uniform(min_vals, max_vals)
            else:
                new_C[i] = np.mean(points, axis=0)

        if np.all(C == new_C):
            return C, clss

        C = new_C

    # Recompute final cluster assignments based on final centroids
    distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
