#!/usr/bin/env python3
"""Loads the FrozenLake environment"""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from gym

    Args:
        desc: custom map description (list of lists)
        map_name: name of a pre-made map
        is_slippery: whether the ice is slippery

    Returns:
        env: the FrozenLake environment
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )

    return env
