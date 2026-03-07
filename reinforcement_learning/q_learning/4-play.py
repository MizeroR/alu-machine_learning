#!/usr/bin/env python3
"""Play one episode with a trained Q-table"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Runs a trained agent on an environment using the Q-table

    Args:
        env: FrozenLake environment
        Q: Q-table
        max_steps: maximum steps for the episode

    Returns:
        total reward for the episode
    """

    total_rewards = 0
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    for _ in range(max_steps):

        # Choose best action (exploit)
        action = np.argmax(Q[state])

        # Take step
        step = env.step(action)
        if len(step) == 5:
            new_state, reward, terminated, truncated, _ = step
            done = terminated or truncated
        else:
            new_state, reward, done, _ = step

        # Render the board with current position
        desc = env.desc.copy().astype(str)
        row, col = divmod(state, env.ncol)
        desc[row, col] = f'`{desc[row, col]}`'
        for r in desc:
            print(''.join(r))
        moves = ['Left', 'Down', 'Right', 'Up']
        print('  ({})'.format(moves[action]))
        print()

        total_rewards += reward
        state = new_state

        if done:
            break

    return total_rewards
