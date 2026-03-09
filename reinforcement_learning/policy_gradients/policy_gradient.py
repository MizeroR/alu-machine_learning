#!/usr/bin/env python3
"""Policy gradient - simple policy function"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy using a weight matrix

    Args:
        matrix: numpy.ndarray of shape (1, n)
        weight: numpy.ndarray of shape (n, m)

    Returns:
        Policy (probabilities) as numpy.ndarray of shape (1, m)
    """

    z = np.dot(matrix, weight)

    # Softmax (numerically stable)
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)
