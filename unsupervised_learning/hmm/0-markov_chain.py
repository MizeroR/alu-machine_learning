#!/usr/bin/env python3
"""
Determines the probability of a Markov chain being in a
particular state after a specified number of iterations
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Calculates the probability of a Markov chain after t iterations

    Parameters:
    P: numpy.ndarray of shape (n, n)
        Transition matrix
    s: numpy.ndarray of shape (1, n)
        Initial state probabilities
    t: int
        Number of iterations

    Returns:
    numpy.ndarray of shape (1, n) of state probabilities after t
    or None on failure
    """

    # Validate P
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None

    # Validate s
    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if not np.allclose(np.sum(s), 1):
        return None

    # Validate t
    if not isinstance(t, int) or t < 0:
        return None

    # Compute P^t
    Pt = np.linalg.matrix_power(P, t)

    # Final state distribution
    return np.matmul(s, Pt)
