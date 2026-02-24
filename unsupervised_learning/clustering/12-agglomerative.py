#!/usr/bin/env python3
"""Performs Agglomerative clustering"""

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage

    Parameters:
    X: numpy.ndarray of shape (n, d)
    dist: maximum cophenetic distance for clusters

    Returns:
    clss: numpy.ndarray of shape (n,)
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(dist, (int, float)) or
            dist < 0):
        return None

    # Ward linkage
    Z = sch.linkage(X, method='ward')

    # Form flat clusters using maximum distance
    clss = sch.fcluster(Z, t=dist, criterion='distance')

    # Plot dendrogram
    plt.figure()
    sch.dendrogram(Z, color_threshold=dist)
    plt.axhline(y=dist, c='k')
    plt.show()

    return clss
