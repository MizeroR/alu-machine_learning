#!/usr/bin/env python3
"""
Calculates the most likely sequence of hidden states for a Hidden Markov Model
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Performs the Viterbi algorithm.

    Parameters
    ----------
    Observation : numpy.ndarray of shape (T,)
        Index of the observation.
    Emission : numpy.ndarray of shape (N, M)
        Emission probability matrix.
    Transition : numpy.ndarray of shape (N, N)
        Transition probability matrix.
    Initial : numpy.ndarray of shape (N, 1)
        Initial state probabilities.

    Returns
    -------
    path : list of length T
        Most likely sequence of hidden states.
    P : float
        Probability of obtaining the path sequence.
    """
    if (not isinstance(Observation, np.ndarray) or
        not isinstance(Emission, np.ndarray)):
        return None, None

    if (not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    N = Emission.shape[0]  # number of hidden states
    T = Observation.shape[0]  # number of observations

    # Initialize DP and backpointer matrices
    V = np.zeros((N, T))  # Viterbi probabilities
    B = np.zeros((N, T), dtype=int)  # Backpointers

    # Initialization step
    V[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Recursion step
    for t in range(1, T):
        for s in range(N):
            prob = V[:, t - 1] * Transition[:, s] * Emission[s, Observation[t]]
            B[s, t] = np.argmax(prob)
            V[s, t] = np.max(prob)

    # Termination step
    P = np.max(V[:, -1])
    path = [int(np.argmax(V[:, -1]))]

    # Backtrack to find the full path
    for t in range(T - 1, 0, -1):
        path.insert(0, B[path[0], t])

    return path, P
