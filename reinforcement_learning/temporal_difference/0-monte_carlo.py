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
        state = env.reset()
        # Handle gymnasium API returning (state, info)
        if isinstance(state, tuple):
            state = state[0]
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)
            
            # Handle both APIs: gym (4) vs gymnasium (5)
            if len(step_result) == 5:
                new_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                new_state, reward, done, _ = step_result
            
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
