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
