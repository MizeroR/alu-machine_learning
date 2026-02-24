#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular Markov chain
"""

import numpy as np


def regular(P):
    """
    Calculates the steady state probabilities of a regular Markov chain

    Parameters:
    P: numpy.ndarray of shape (n, n)
        Transition matrix

    Returns:
    numpy.ndarray of shape (1, n) containing steady state probabilities
    or None on failure
    """

    # Validate P
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    n, m = P.shape
    if n != m:
        return None

    # Rows must sum to 1
    if not np.allclose(np.sum(P, axis=1), 1):
        return None

    # Check regularity:
    # A Markov chain is regular if some power of P has all positive entries
    Pk = np.copy(P)
    for _ in range(1, n * n + 1):
        if np.all(Pk > 0):
            break
        Pk = np.matmul(Pk, P)
    else:
        return None

    # Solve for steady state π such that πP = π
    # Equivalent to solving (P^T - I)π^T = 0 with sum(π) = 1

    A = P.T - np.eye(n)
    A[-1] = np.ones(n)  # replace last equation with sum constraint
    b = np.zeros(n)
    b[-1] = 1

    try:
        steady = np.linalg.solve(A, b)
    except Exception:
        return None

    return steady.reshape(1, n)
