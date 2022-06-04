import gym 
import turtlebot_env
import pybullet as p
import time

env = gym.make('Turtlebot-v3', use_gui=False) 
observation = env.reset()
print(observation)
# actions = []
# for i in range(5):
#     actions.append(env.action_space.sample())

# observation, reward, done, info = env.step(actions)
