#!/usr/bin/env python3
"""
Poisson distribution class without external imports.
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF for a given number of successes.

        Args:
            k: number of successes (int or convertible to int)

        Returns:
            PMF value for k (float)
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        return (self._exp(-self.lambtha) *
                (self.lambtha ** k)) / self._factorial(k)

    def _factorial(self, n):
        """
        Computes factorial of n.
        """
        if n == 0 or n == 1:
            return 1
        fact = 1
        for i in range(2, n + 1):
            fact *= i
        return fact

    def _exp(self, x):
        """
        Computes the exponential of x using a Taylor series approximation.
        """
        result = 1
        term = 1
        for i in range(1, 50):  # 50 terms for good accuracy
            term *= x / i
            result += term
        return result
