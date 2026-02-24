#!/usr/bin/env python3
"""
Performs the Forward algorithm for a Hidden Markov Model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a Hidden Markov Model.

    Observation: numpy.ndarray of shape (T,) with observation indices
    Emission: numpy.ndarray of shape (N, M), emission probabilities
    Transition: numpy.ndarray of shape (N, N), transition probabilities
    Initial: numpy.ndarray of shape (N, 1), initial state probabilities

    Returns: P, F
        P: likelihood of the observation sequence
        F: numpy.ndarray of shape (N, T), forward path probabilities
    """
    if (not isinstance(Observation, np.ndarray) or
            not isinstance(Emission, np.ndarray) or
            not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    T = Observation.shape[0]  # number of observations
    N = Emission.shape[0]     # number of hidden states

    # Initialize forward probability matrix
    F = np.zeros((N, T))

    # Base case: t = 0
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Recursive case: t > 0
    for t in range(1, T):
        for j in range(N):
            F[j, t] = (Emission[j, Observation[t]] *
                       np.sum(F[:, t-1] * Transition[:, j]))

    # Total probability of observation sequence
    P = np.sum(F[:, -1])

    return P, F
