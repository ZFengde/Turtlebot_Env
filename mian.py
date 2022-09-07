import gym 
import turtlebot_env
import pybullet as p
import time

env = gym.make('Turtlebot-v2', use_gui=True) 

actions = []
while True:
    observation = env.reset()
    while True:
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        print(obs.shape)
        time.sleep(1./240.)
        if done:
            break
