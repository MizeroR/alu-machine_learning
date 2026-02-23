#!/usr/bin/env python3
"""Expectation Maximization for a GMM"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """
    Performs the EM algorithm for a GMM

    Returns:
        pi, m, S, g, log_l
    or
        None, None, None, None, None on failure
    """

    # Validate inputs
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None, None, None, None, None

    if (not isinstance(k, int) or k <= 0):
        return None, None, None, None, None

    if (not isinstance(iterations, int) or iterations <= 0):
        return None, None, None, None, None

    if (not isinstance(tol, float) or tol < 0):
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialize parameters
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    # Initial expectation
    g, log_l_prev = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(log_l_prev))

    # ONE LOOP (allowed)
    for i in range(1, iterations + 1):

        # Maximization
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        # Expectation
        g, log_l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        # Verbose printing
        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(i, log_l))

        # Early stopping
        if abs(log_l - log_l_prev) <= tol:
            if verbose and i % 10 != 0:
                print("Log Likelihood after {} iterations: {:.5f}"
                      .format(i, log_l))
            break

        log_l_prev = log_l

    return pi, m, S, g, log_l
