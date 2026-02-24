#!/usr/bin/env python3
"""Gaussian Mixture Model using sklearn"""

import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset

    Parameters:
    X: numpy.ndarray of shape (n, d)
    k: number of clusters

    Returns:
    pi: shape (k,)
    m: shape (k, d)
    S: shape (k, d, d)
    clss: shape (n,)
    bic: float
    """

    if (not hasattr(X, "shape") or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0):
        return None, None, None, None, None

    model = sklearn.mixture.GaussianMixture(n_components=k)

    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
