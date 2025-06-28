#!/usr/bin/env python3
"""
Defines an Binomial distribution class without external modules.
"""


class Binomial:
    """
    Represents a binomial distribution.

    Attributes:
        n (int): Number of Bernoulli trials.
        p (float): Probability of a "success".
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        Args:
            data (list, optional): Data to estimate n and p.
            n (int, optional): Number of Bernoulli trials.
            p (float, optional): Probability of a "success".

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than two data points.
            ValueError: If n is not positive.
            ValueError: If p is not a valid probability.
        """
        if data is None:
            # Use given n and p
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Calculate n and p from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance from data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # For binomial distribution: mean = n*p, variance = n*p*(1-p)
            # From these: p = 1 - (variance/mean), n = mean/p

            # Calculate p first
            p_estimated = 1 - (variance / mean)

            # Calculate n
            n_estimated = mean / p_estimated

            # Round n to nearest integer
            self.n = round(n_estimated)

            # Recalculate p using the rounded n
            self.p = mean / self.n

    def _factorial(self, num):
        """Returns the factorial of a number"""
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result

    def _comb(self, n, k):
        """Returns the number of combinations of n items taken k at a time"""
        return self._factorial(n) // (self._factorial(k) *
                                      self._factorial(n - k))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”.

        Args:
            k (int): Number of “successes”

        Returns:
            float: PMF value for k
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        return self._comb(self.n, k) * (self.p ** k) * ((1 - self.p) **
                                                        (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”.

        Args:
            k (int): Number of “successes”

        Returns:
            float: CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        cdf_value = 0
        for i in range(0, k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
