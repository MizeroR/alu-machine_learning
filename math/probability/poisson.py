#!/usr/bin/env python3
"""
This module defines a Poisson class that represents a Poisson distribution.
"""


import math


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution.

        Parameters:
        - data (list): Optional list of data to estimate the distribution.
        - lambtha (float): Expected number of occurrences in a given time frame.

        Raises:
        - TypeError: If data is not a list.
        - ValueError: If data contains fewer than two values or
                      if lambtha is not a positive value.
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
            k (int or float): Number of successes.

        Returns:
            float: PMF value for k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        lambtha = self.lambtha
        return (math.exp(-lambtha) * (lambtha ** k)) / math.factorial(k)
