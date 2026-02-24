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
    if not isinstance(Observations, np.ndarray) or len(Observations.shape) != 1:
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
        # Forward algorithm
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            for j in range(M):
                F[j, t] = (Emission[j, Observations[t]] *
                           np.sum(F[:, t-1] * Transition[:, j]))
        P = np.sum(F[:, -1])

        # Backward algorithm
        B = np.zeros((M, T))
        B[:, -1] = 1
        for t in range(T - 2, -1, -1):
            for i in range(M):
                B[i, t] = np.sum(Transition[i, :] *
                                 Emission[:, Observations[t + 1]] *
                                 B[:, t + 1])

        # Compute xi and gamma
        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = (F[i, t] * Transition[i, j] *
                                   Emission[j, Observations[t + 1]] *
                                   B[j, t + 1])
            xi[:, :, t] /= (P if P > 0 else 1e-12)

        for t in range(T):
            gamma[:, t] = (F[:, t] * B[:, t]) / (P if P > 0 else 1e-12)

        # Re-estimate parameters
        for i in range(M):
            denom = gamma[i, :-1].sum()
            if denom == 0:
                denom = 1e-12
            Transition[i, :] = xi[i, :, :].sum(axis=1) / denom

        for i in range(M):
            for k in range(N):
                mask = (Observations == k)
                denom = gamma[i, :].sum()
                if denom == 0:
                    denom = 1e-12
                Emission[i, k] = gamma[i, mask].sum() / denom

    return Transition, Emission
