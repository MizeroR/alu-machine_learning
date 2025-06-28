#!/usr/bin/env python3
"""
Module: 3-posterior
Calculates the posterior probability of each hypothetical P given the data.
"""

import numpy as np  # type: ignore
likelihood = __import__('0-likelihood').likelihood
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data.

    Parameters:
    - x (int): number of patients with severe side effects
    - n (int): total number of patients
    - P (np.ndarray): 1D array of hypothetical probabilities
    - Pr (np.ndarray): 1D array of prior beliefs of P

    Returns:
    - np.ndarray: posterior probability for each probability in P
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

    # Posterior = (Likelihood * Prior) / Marginal
    likelihood_vals = likelihood(x, n, P)
    marginal_val = marginal(x, n, P, Pr)
    posterior_vals = likelihood_vals * Pr / marginal_val

    return posterior_vals
