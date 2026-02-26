#!/usr/bin/env python3
"""TD(lambda) algorithm"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm"""

    if not isinstance(V, np.ndarray):
        return None

    n_states = V.shape[0]

    for _ in range(episodes):
        state, _ = env.reset()
        E = np.zeros(n_states)

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = \
                env.step(action)

            done = terminated or truncated

            delta = reward + gamma * V[new_state] - V[state]

            E[state] += 1

            V += alpha * delta * E

            E *= gamma * lambtha

            state = new_state

            if done:
                break

    return V
