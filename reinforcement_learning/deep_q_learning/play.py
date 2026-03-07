#!/usr/bin/env python3
"""
Play Breakout using a trained DQN agent
"""

import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

# Create environment
env = gym.make("Breakout-v0")
np.random.seed(0)
env.seed(0)
nb_actions = env.action_space.n

# Build the same model architecture as training
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                 input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Setup memory and policy
memory = SequentialMemory(limit=50000, window_length=1)
policy = GreedyQPolicy()

# Setup DQN agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=1, target_model_update=1e-2, policy=policy)
dqn.compile('adam', metrics=['mae'])

# Load the trained weights
dqn.load_weights("policy.h5")

# Play one episode
dqn.test(env, nb_episodes=1, visualize=True)
