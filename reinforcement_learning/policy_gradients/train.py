#!/usr/bin/env python3
"""Train using Policy Gradient (REINFORCE)"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045,
          gamma=0.98, show_result=False):
    """
    Implements policy gradient training

    Args:
        env: initial environment
        nb_episodes: number of episodes
        alpha: learning rate
        gamma: discount factor
        show_result: render every 1000 episodes if True

    Returns:
        scores: list of total rewards per episode
    """

    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )

    scores = []

    for episode in range(nb_episodes):

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = state[None, :]

        grads = []
        rewards = []
        score = 0

        for _ in range(1000):

            # Render every 1000 episodes
            if show_result and (episode + 1) % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)

            step = env.step(action)
            if len(step) == 5:
                new_state, reward, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                new_state, reward, done, _ = step

            grads.append(grad)
            rewards.append(reward)
            score += reward

            state = new_state[None, :]

            if done:
                break

        scores.append(score)

        # Compute discounted rewards
        discounted = np.zeros_like(rewards, dtype=float)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            discounted[t] = G

        # Normalize rewards
        if np.std(discounted) != 0:
            discounted = (discounted - np.mean(discounted)) / np.std(discounted)

        # Update weights
        for grad, Gt in zip(grads, discounted):
            weight += alpha * grad * Gt

        print(f"Episode {episode + 1}: {score}", end="\r", flush=False)

    return scores
