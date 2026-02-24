#!/usr/bin/env python3
"""Find best number of clusters for a GMM using BIC"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Determines the best number of clusters for a GMM using BIC

    Returns:
        best_k, best_result, log_l, b
    or
        None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = range(kmin, kmax + 1)
    log_l_values = []
    b_values = []
    results = []

    # ONE LOOP (allowed)
    for k in ks:
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )
        if pi is None:
            return None, None, None, None

        # Number of parameters in a GMM:
        # p = k-1 (weights) + k*d (means) + k*d*(d+1)/2 (covariances)
        p = (k - 1) + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * log_l

        log_l_values.append(log_l)
        b_values.append(bic)
        results.append((pi, m, S))

    # Convert to numpy arrays
    log_l_values = np.array(log_l_values)
    b_values = np.array(b_values)

    # Best k is one with minimum BIC
    best_idx = np.argmin(b_values)
    best_k = ks[best_idx]
    best_result = results[best_idx]

    return best_k, best_result, log_l_values, b_values
