import gym 
import turtlebot_env
import pybullet as p
import time

env = gym.make('Turtlebot-v2', use_gui=True) 

while True:
    observation = env.reset()
    for t in range(5000):
        # env.render()
        action = env.action_space.sample()
        # observation = xy position, xy orientation, xy direction velocity, target xy position
        observation, reward, done, obstacle_info = env.step(action)
        print(reward)
        # position = observation[:2]
        # orientation = observation[2:4]
        # velocity = observation[4:6]
        # target_position = observation[6:]
        # time.sleep(1./240.)
        if done:
            break
