#!/usr/bin/env python3
"""Backward algorithm for a Hidden Markov Model."""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a Hidden Markov Model.

    Observation: np.ndarray of shape (T,) with observation indices
    Emission: np.ndarray of shape (N, M) with emission probabilities
    Transition: np.ndarray of shape (N, N) with transition probabilities
    Initial: np.ndarray of shape (N, 1) with initial probabilities

    Returns:
        P: likelihood of the observations given the model
        B: np.ndarray of shape (N, T) containing backward probabilities
    """
    if (not isinstance(Observation, np.ndarray) or
            not isinstance(Emission, np.ndarray)):
        return None, None
    if (not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    N = Emission.shape[0]
    T = Observation.shape[0]

    # Initialize B
    B = np.zeros((N, T))
    B[:, -1] = 1  # Last column = 1

    # Iterate backwards
    for t in range(T - 2, -1, -1):
        B_next = B[:, t + 1] * Emission[:, Observation[t + 1]]
        B[:, t] = np.dot(Transition, B_next)
    # Compute total probability
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
