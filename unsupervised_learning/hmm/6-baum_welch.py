#!/usr/bin/env python3
"""
6-baum_welch.py
Baum-Welch algorithm for Hidden Markov Models
"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.

    Arguments:
        Observations: numpy.ndarray of shape (T,) with observation indices
        Transition: numpy.ndarray of shape (M, M) transition probabilities
        Emission: numpy.ndarray of shape (M, N) emission probabilities
        Initial: numpy.ndarray of shape (M, 1) initial state probabilities
        iterations: number of EM iterations to perform (default 1000)

    Returns:
        Transition: the converged transition probabilities
        Emission: the converged emission probabilities
        or None, None on failure
    """
    if (not isinstance(Observations, np.ndarray)
            or len(Observations.shape) != 1):
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    T = Observations.shape[0]
    M = Transition.shape[0]
    N = Emission.shape[1]

    if Transition.shape[1] != M or Emission.shape[0] != M:
        return None, None
    if Initial.shape[0] != M or Initial.shape[1] != 1:
        return None, None

    for _ in range(iterations):
        # Forward algorithm (vectorized)
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = Emission[:, Observations[t]] * (Transition.T @ F[:, t-1])
        P = np.sum(F[:, -1])
        if P == 0:
            P = 1e-12

        # Backward algorithm (vectorized)
        B = np.zeros((M, T))
        B[:, -1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = Transition @ (Emission[:, Observations[t + 1]] * B[:, t + 1])

        # Compute xi and gamma
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            xi[:, :, t] = (F[:, t, np.newaxis] * Transition * 
                          Emission[:, Observations[t + 1]][np.newaxis, :] * 
                          B[:, t + 1][np.newaxis, :])
            xi[:, :, t] /= P

        gamma = (F * B) / P

        # Re-estimate Transition
        Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1, keepdims=True)

        # Re-estimate Emission
        for k in range(N):
            mask = (Observations == k)
            Emission[:, k] = np.sum(gamma[:, mask], axis=1) / np.sum(gamma, axis=1)

    return Transition, Emission
