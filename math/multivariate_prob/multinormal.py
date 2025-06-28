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
