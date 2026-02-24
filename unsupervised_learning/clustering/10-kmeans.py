#!/usr/bin/env python3
"""Performs K-means clustering using sklearn"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset

    Parameters:
    X: numpy.ndarray of shape (n, d)
    k: number of clusters

    Returns:
    C: numpy.ndarray of shape (k, d) -> centroids
    clss: numpy.ndarray of shape (n,) -> cluster indices
    """

    if (not hasattr(X, "shape") or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0):
        return None, None

    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)

    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
