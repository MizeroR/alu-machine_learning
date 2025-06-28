#!/usr/bin/env python3
"""
Calculates the likelihood of obtaining data given various probabilities.
"""
import numpy as np  # type: ignore


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data x and n
    for each probability in P.

    Parameters:
    - x: number of patients with severe side effects
    - n: total number of patients
    - P: 1D numpy.ndarray of hypothetical probabilities

    Returns:
    - 1D numpy.ndarray of likelihoods for each probability in P
    """
    # Input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    def factorial(n):
        """Returns the factorial of n."""
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    # Likelihood calculation
    binom_coeff = factorial(n) // (factorial(x) * factorial(n - x))
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods.astype(float)
