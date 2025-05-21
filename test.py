
import gymnasium as gym
import numpy as np
import turtlebot_env

env = gym.make("Turtlebot-v3")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(terminated, truncated, info)
    episode_over = terminated or truncated
