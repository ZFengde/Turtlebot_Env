import gym 
import turtlebot_env
import pybullet as p
import time

env = gym.make('Turtlebot-v3', use_gui=True) 

actions = []
while True:
    obs = env.reset()
    while True:
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        if 'cost' in info.keys():
            print(info['cost'])
        time.sleep(1./480.)
        if done:
            break
