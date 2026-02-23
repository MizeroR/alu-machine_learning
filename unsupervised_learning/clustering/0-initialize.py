#!/usr/bin/env python3
"""Module for initializing K-means centroids"""


import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Parameters:
    X (numpy.ndarray): dataset of shape (n, d)
    k (int): number of clusters

    Returns:
    numpy.ndarray: initialized centroids of shape (k, d)
    None: on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0):
        return None

    n, d = X.shape

    if k > n:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(k, d)
    )

    return centroids
