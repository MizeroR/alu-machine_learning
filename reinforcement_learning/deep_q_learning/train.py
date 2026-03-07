#!/usr/bin/env python3
"""
Train a DQN agent to play Atari Breakout
"""

import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Create the environment
env = gym.make("Breakout-v0")
np.random.seed(0)
env.seed(0)
nb_actions = env.action_space.n

# Build a simple CNN model
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                 input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

# Setup memory and policy
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()

# Setup DQN agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Save the trained weights
dqn.save_weights("policy.h5", overwrite=True)
