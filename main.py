import gym 
import turtlebot_env
import pybullet as p
import time
import numpy as np

env = gym.make('Turtlebot-v2', use_gui=True) 

actions = []
while True:
    obs = env.reset()
    returns = 0
    costs = 0
    while True:
        actions = np.array([1, 1])
        obs, reward, done, info = env.step(actions)
        returns += reward
        if 'cost' in info.keys():
            costs += info['cost']
        time.sleep(1./480.)
        if done:
            print(returns, costs)
            returns = 0
            break
