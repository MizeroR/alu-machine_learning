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

def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient

    Args:
        state: current observation (1, n)
        weight: weights matrix (n, m)

    Returns:
        action, gradient
    """

    probs = policy(state, weight)

    # Sample action from probability distribution
    action = np.random.choice(len(probs[0]), p=probs[0])

    # One-hot encoding of action
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1

    # Gradient: state^T * (one_hot - probs)
    grad = np.dot(state.T, (one_hot - probs))

    return action, grad
