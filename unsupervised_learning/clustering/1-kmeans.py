#!/usr/bin/env python3
"""Module for performing K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset

    Parameters:
    X (numpy.ndarray): dataset of shape (n, d)
    k (int): number of clusters
    iterations (int): maximum number of iterations

    Returns:
    C (numpy.ndarray): centroid means of shape (k, d)
    clss (numpy.ndarray): cluster indices of shape (n,)
    (None, None): on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0 or
            not isinstance(iterations, int) or
            iterations <= 0):
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    # Initialize centroids (uniform distribution)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, (k, d))

    for _ in range(iterations):
        # Compute distances
        distances = np.linalg.norm(
            X[:, np.newaxis] - C, axis=2
        )

        # Assign clusters
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        # Update centroids
        for i in range(k):
            points = X[clss == i]

            if points.shape[0] == 0:
                new_C[i] = np.random.uniform(
                    min_vals, max_vals
                )
            else:
                new_C[i] = np.mean(points, axis=0)

        # Check for convergence
        if np.all(C == new_C):
            return C, clss

        C = new_C

    return C, clss
