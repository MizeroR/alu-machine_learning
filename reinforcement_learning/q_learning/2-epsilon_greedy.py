#!/usr/bin/env python3
"""Epsilon-greedy action selection"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using epsilon-greedy

    Args:
        Q: numpy.ndarray containing the Q-table
        state: current state
        epsilon: probability of exploring

    Returns:
        action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
