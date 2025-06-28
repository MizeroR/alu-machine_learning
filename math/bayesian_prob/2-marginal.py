#!/usr/bin/env python3
"""
Module: 2-marginal
Calculates the marginal probability of obtaining the data.
"""

import numpy as np  # type: ignore
likelihood = __import__('0-likelihood').likelihood


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data x and n.

    Parameters:
    - x (int): number of patients with severe side effects
    - n (int): total number of patients
    - P (np.ndarray): 1D array of hypothetical probabilities
    - Pr (np.ndarray): 1D array of prior beliefs of P

    Returns:
    - float: marginal probability of observing x out of n
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute marginal: sum of intersection (likelihood * prior)
    return np.sum(likelihood(x, n, P) * Pr)
