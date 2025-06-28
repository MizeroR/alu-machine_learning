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
        Calculates the value of the CDF for a given x-value.
        Args:
            x (float): The x-value.
        Returns:
            float: The CDF value for x.
        """
        # CDF formula using error function approximation
        # CDF(x) = 0.5 * (1 + erf((x - μ) / (σ * √2)))

        # Calculate z-score normalized by √2
        z = (x - self.mean) / self.stddev

        if z < 0:
            return 1 - self.cdf(2 * self.mean - x)

        # Error function approximation using Taylor series
        # erf(z) ≈ (2/√π) * (z - z³/3 + z⁵/10 - z⁷/42 + z⁹/216 - ...)
        pi = 3.1415926536
        e = 2.7182818285

        # For better accuracy, use more terms in the series
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        k = 1 / (1 + 0.2316419 * z)

        pdf_z = (1 / (2 * pi) ** 0.5) * (e ** (-0.5 * z * z))

        cdf_z = 1 - pdf_z * (a1 * k + a2 *
                             k**2 + a3 * k**3 + a4 *
                             k**4 + a5 * k**5
                            )

        return cdf_z
