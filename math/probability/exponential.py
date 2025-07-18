#!/usr/bin/env python3
"""
Defines an Exponential distribution class without external modules.
"""


class Exponential:
    """
    Represents an exponential distribution.

    Attributes:
        lambtha (float): The number of occurrences in a given time.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.

        Args:
            data (list, optional): List of data to estimate the distribution.
            lambtha (float, optional): Number of occurrences in a time frame.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
            ValueError: If lambtha is not a positive value.
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
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period x.

        Args:
            x (float): The time period.

        Returns:
            float: PDF value for x, or 0 if x is out of range.
        """
        if x < 0:
            return 0
        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period x.

        Args:
            x (float): The time period.

        Returns:
            float: CDF value for x, or 0 if x is out of range.
        """
        if x < 0:
            return 0
        return 1 - (2.7182818285 ** (-self.lambtha * x))
