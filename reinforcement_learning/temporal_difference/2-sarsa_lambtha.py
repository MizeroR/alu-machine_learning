#!/usr/bin/env python3
"""SARSA(lambda) algorithm"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Performs the SARSA(lambda) algorithm"""

    if not isinstance(Q, np.ndarray):
        return None

    n_states, n_actions = Q.shape

    def epsilon_greedy(state, eps):
        """Select action using epsilon-greedy"""
        if np.random.uniform() < eps:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])

    for _ in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(state, epsilon)

        E = np.zeros((n_states, n_actions))

        for _ in range(max_steps):
            new_state, reward, terminated, truncated, _ = \
                env.step(action)

            done = terminated or truncated

            new_action = epsilon_greedy(new_state, epsilon)

            delta = reward + gamma * Q[new_state, new_action] \
                - Q[state, action]

            E[state, action] += 1

            Q += alpha * delta * E

            E *= gamma * lambtha

            state = new_state
            action = new_action

            if done:
                break

        epsilon = max(min_epsilon,
                      epsilon * (1 - epsilon_decay))

    return Q
