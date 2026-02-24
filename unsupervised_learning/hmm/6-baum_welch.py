#!/usr/bin/env python3
"""
6-baum_welch.py
Baum-Welch algorithm for Hidden Markov Models
"""

import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.

    Arguments:
        Observations: numpy.ndarray of shape (T,) with the observation indices
        Transition: numpy.ndarray of shape (M, M) initialized transition probabilities
        Emission: numpy.ndarray of shape (M, N) initialized emission probabilities
        Initial: numpy.ndarray of shape (M, 1) initial hidden state probabilities
        iterations: number of EM iterations to perform (default 1000)

    Returns:
        Transition: the converged transition probabilities
        Emission: the converged emission probabilities
        or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or not isinstance(Transition, np.ndarray):
        return None, None
    if not isinstance(Emission, np.ndarray) or not isinstance(Initial, np.ndarray):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    T = Observations.shape[0]
    M = Transition.shape[0]
    N = Emission.shape[1]

    for _ in range(iterations):
        P, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        # Compute xi[i,j,t] = P(state i at t, state j at t+1 | observations)
        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = (F[i, t] * Transition[i, j] *
                                   Emission[j, Observations[t + 1]] *
                                   B[j, t + 1])
            xi[:, :, t] /= (P if P > 0 else 1e-12)

        # Compute gamma[i,t] = P(state i at t | observations)
        for t in range(T):
            gamma[:, t] = (F[:, t] * B[:, t]) / (P if P > 0 else 1e-12)

        # Re-estimate Transition
        for i in range(M):
            denom = gamma[i, :-1].sum()
            if denom == 0:
                denom = 1e-12
            Transition[i, :] = xi[i, :, :].sum(axis=1) / denom

        # Re-estimate Emission
        for i in range(M):
            for k in range(N):
                mask = (Observations == k)
                denom = gamma[i, :].sum()
                if denom == 0:
                    denom = 1e-12
                Emission[i, k] = gamma[i, mask].sum() / denom

        # Re-estimate Initial
        Initial[:, 0] = gamma[:, 0]

    return Transition, Emission
