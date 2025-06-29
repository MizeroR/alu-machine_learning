#!/usr/bin/env python3
"""
Defines a Poisson distribution class with no external modules.
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution.

        Args:
            data (list): list of observed data (optional)
            lambtha (float): expected number of occurrences in a time frame

        Raises:
            TypeError: If data is not a list
            ValueError: If lambtha <= 0 or data is too short
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
        Calculates the value of the PMF for a given number of “successes”.

        Args:
            k (int): The number of occurrences.

        Returns:
            float: PMF value for k.
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        # Calculate k!
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        e = 2.7182818285
        lambtha = self.lambtha

        return ((lambtha ** k) * (e ** -lambtha)) / factorial

    def cdf(self, k):
        """
        Calculates the CDF value for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: CDF value for k.
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
