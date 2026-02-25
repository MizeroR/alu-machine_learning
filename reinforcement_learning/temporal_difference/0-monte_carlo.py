#!/usr/bin/env python3
"""Monte Carlo method for value estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.9):
    """
    Performs the Monte Carlo algorithm

    Parameters:
    - env: OpenAI Gym environment
    - V: numpy.ndarray of shape (s,) containing value estimates
    - policy: function mapping state -> action
    - episodes: number of training episodes
    - max_steps: max steps per episode
    - alpha: learning rate
    - gamma: discount factor

    Returns:
    - Updated value function V
    """

    for _ in range(episodes):
        result = env.reset()
        # Handle both gym and gymnasium API
        if isinstance(result, tuple):
            state = result[0]
        else:
            state = result
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            result = env.step(action)
            # Handle both gym (4 returns) and gymnasium (5 returns)
            if len(result) == 5:
                new_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                new_state, reward, done, _ = result
            episode.append((state, reward))
            state = new_state
            if done:
                break

        # Compute returns and update V (backward)
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = gamma * G + r

            # First-visit Monte Carlo
            if s not in visited:
                visited.add(s)
                V[s] = V[s] + alpha * (G - V[s])

    return V
