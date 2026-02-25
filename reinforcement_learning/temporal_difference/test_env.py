#!/usr/bin/env python3
import gymnasium as gym
import numpy as np

# Test with deterministic environment
env = gym.make('FrozenLake8x8-v1', is_slippery=False)
state, _ = env.reset(seed=0)
print(f"Initial state: {state}")
print(f"Environment desc:\n{env.unwrapped.desc}")
print(f"Goal position: {env.unwrapped.desc == b'G'}")

# Try to reach goal deterministically
actions = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]  # Down and Right moves
total_reward = 0
for i, action in enumerate(actions):
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"Step {i}: action={action}, state={state}, reward={reward}, done={terminated or truncated}")
    if terminated or truncated:
        print(f"Episode ended with total reward: {total_reward}")
        break
