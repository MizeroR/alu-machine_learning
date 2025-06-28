#!/usr/bin/env python3
"""
Defines the MultiNormal class for a Multivariate Normal distribution.
"""

import numpy as np  # type: ignore


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the distribution with the given data.

        Args:
            data (numpy.ndarray): of shape (d, n), dataset with:
                - d: number of dimensions
                - n: number of data points

        Raises:
            TypeError: if data is not a 2D numpy.ndarray
            ValueError: if n < 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Compute mean (shape (d, 1))
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data and compute covariance matrix manually
        X_centered = data - self.mean
        self.cov = np.dot(X_centered, X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the value of the PDF at a given data point.

        Args:
            x (numpy.ndarray): shape (d, 1), the data point

        Returns:
            float: the value of the PDF at x

        Raises:
            TypeError: if x is not a numpy.ndarray
            ValueError: if x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        pi = np.pi

        exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
        denominator = np.sqrt(((2 * pi) ** d) * det_cov)
        result = (1. / denominator) * np.exp(exponent)

        return result.item()
