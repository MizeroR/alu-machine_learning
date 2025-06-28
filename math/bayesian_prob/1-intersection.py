#!/usr/bin/env python3
"""Calculates the intersection of the likelihood and prior beliefs."""
import numpy as np  # type: ignore


def factorial(k):
    """
    Calculates the factorial of a non-negative integer.

    Parameters:
        k (int): A non-negative integer whose factorial is to be computed.

    Returns:
        int: The factorial of k (i.e., k!).

    Raises:
        ValueError: If k is not an integer or is negative.

    Example:
        factorial(5)  # returns 120 because 5! = 5 × 4 × 3 × 2 × 1
    """
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result


def likelihood(x, n, P):
    """Calculate the likelihood of x successes in n trials for all P."""
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
    if not np.all((0 <= P) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    binom_coeff = factorial(n) // (factorial(x) * factorial(n - x))
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))
    return np.array(likelihoods, dtype=np.float64)


def intersection(x, n, P, Pr):
    """Calculate intersection of data likelihood and prior beliefs."""
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
    if not np.all((0 <= P) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((0 <= Pr) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    like = likelihood(x, n, P)
    inter = like * Pr
    return np.array(inter, dtype=np.float64)
