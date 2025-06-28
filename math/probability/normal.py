#!/usr/bin/env python3
"""
Module that defines a Normal distribution class.
"""


class Normal:
    """
    Represents a normal distribution.

    Attributes:
        mean (float): Mean of the distribution.
        stddev (float): Standard deviation of the distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution.

        Args:
            data (list, optional): Data to estimate mean and stddev.
            mean (float, optional): Mean of the distribution.
            stddev (float, optional): Deviation of the distribution.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than two data points.
            ValueError: If stddev is not positive.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            self.mean = float(sum(data) / n)
            variance = sum((x - self.mean) ** 2 for x in data) / n
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.
        Args:
            x (float): The x-value.
        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.
        Args:
            z (float): The z-score.
        Returns:
            float: The x-value of z.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.
        Args:
            x (float): The x-value.
        Returns:
            float: The PDF value for x.
        """
        # PDF formula: (1 / (σ * √(2π))) * e^(-0.5 * ((x - μ) / σ)^2)
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2

        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF (cumulative distribution function)
        for a given x-value in the normal distribution.

        Args:
            x (float): The x-value for which to calculate the CDF.

        Returns:
            float: The probability that a random variable drawn from this
            normal distribution is less than or equal to x.
        """
        z = (x - self.mean) / self.stddev
        if z < 0:
            # Use symmetry property
            return 1 - self.cdf(self.mean - (x - self.mean))

        p = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429

        t = 1 / (1 + p * z)
        pi = 3.141592653589793
        e = 2.718281828459045
        pdf = (1 / (2 * pi) ** 0.5) * (e ** (-z * z / 2))

        poly = b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5

        return 1 - pdf * poly
