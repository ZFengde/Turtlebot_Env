import gym 
import turtlebot_env
import pybullet as p
import time

env = gym.make('Turtlebot-v4', use_gui=True) 

actions = []
while True:
    obs = env.reset()
    while True:
        print(obs)
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        time.sleep(1./240.)
        if done:
            break
