#!/usr/bin/env python3
"""Initializes the Q-table"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    Args:
        env: FrozenLakeEnv instance

    Returns:
        Q-table initialized with zeros
    """
    states = env.observation_space.n
    actions = env.action_space.n

    Q = np.zeros((states, actions))

    return Q
