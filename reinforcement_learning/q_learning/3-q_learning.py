#!/usr/bin/env python3
"""Q-learning training"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """
    Performs Q-learning

    Args:
        env: FrozenLake environment
        Q: Q-table
        episodes: number of episodes
        max_steps: max steps per episode
        alpha: learning rate
        gamma: discount factor
        epsilon: initial epsilon
        min_epsilon: minimum epsilon
        epsilon_decay: epsilon decay rate

    Returns:
        Q, total_rewards
    """

    total_rewards = []

    for _ in range(episodes):

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        rewards = 0

        for _ in range(max_steps):

            action = epsilon_greedy(Q, state, epsilon)

            step = env.step(action)

            if len(step) == 5:
                new_state, reward, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                new_state, reward, done, _ = step

            if done and reward == 0:
                reward = -1

            next_max = np.max(Q[new_state])

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * next_max - Q[state, action]
            )

            state = new_state
            rewards += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        total_rewards.append(rewards)

    return Q, total_rewards
