import gym
import numpy as np
import math
import pybullet as p
from turtlebot_env.resources.turtlebot import turtlebot
from turtlebot_env.resources.plane import plane
from turtlebot_env.resources.target import target

class TurtleBotEnv_Simple(gym.Env):
    metadata = {'render.modes': ['human']}

    # this is for gym environment initialisation
    def __init__(self, use_gui=False):
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        '''
        action space initialisation
        should be the rotation speed omiga here
        radius = 0.033, D = 0.22
        V = (wl+wr)/2, W = (wl - wr)/D
        '''

        # here we should normalisation
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))

        # observation space initialisation
        # observation = xy position[1, 2], xy orientation[3, 4]
        # xy direction velocity[5, 6], target xy position[7, 8]
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-5, -5, -1, -1, -3, -3, -5, -5], dtype=np.float32),
            high=np.array([5, 5, 1, 1, 3, 3, 5, 5], dtype=np.float32))
        
        # this is for random initialisation, could be replaced by another method
        self.np_random, _ = gym.utils.seeding.np_random()

        # placeholders
        self.turtlebot = None
        self.target = None
        self.prev_dist_to_target = None

    # this is what happened in every single step
    def step(self, action):
        # we need manually clip action input into (-1, 1) and then map it into desired velocity
        action = np.tanh(action)
        self.turtlebot.apply_action((action + 1) * 3.25)
        p.stepSimulation()
        turtlebot_ob = self.turtlebot.get_observation()

        # step reward setting, compute L2 distance firstly
        dist_to_target = math.sqrt(((turtlebot_ob[0] - self.target[0]) ** 2 +
                                  (turtlebot_ob[1] - self.target[1]) ** 2))

        # 1. foward reward, 2. time reward
        reward = 10 * (self.prev_dist_to_target - dist_to_target) - 0.01

        self.prev_dist_to_target = dist_to_target

        # 3. Done by running off boundaries penalty
        if (turtlebot_ob[0] >= 2 or turtlebot_ob[0] <= -2 or
                turtlebot_ob[1] >= 2 or turtlebot_ob[1] <= -2):
            self.done = True
            reward = -10

        # 4. Done by reaching target reward
        elif dist_to_target < 0.15:
            self.done = True
            reward = 50

        # + here is actually concantenation
        ob = np.concatenate((turtlebot_ob.reshape(3, 2), 
                                self.target.reshape(1, 2)), dtype=np.float32)

        # To be written
        info = None
        return ob, reward, self.done, info

    # this is for generating random seeds for training
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # this is reset function for initialising each episode
    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        # Reload the plane and car
        plane(self.client)
        self.turtlebot = turtlebot(self.client)
        self.turtlebot_ID, _ = self.turtlebot.get_ids()

        # Set the target to a random target
        x = 1.5
        y = 1.5
        # self.target is the base position of the target
        self.target = np.array((x, y), dtype=float)
        
        self.done = False

        # Visual element of the target
        target(self.client, self.target)

        # Get observation to return
        turtlebot_ob = self.turtlebot.get_observation()

        # this is for generating first prev_dist_to_target when initialising
        # for use in step function
        self.prev_dist_to_target = math.sqrt(((turtlebot_ob[0] - self.target[0]) ** 2 +
                                           (turtlebot_ob[1] - self.target[1]) ** 2))

        return np.concatenate((turtlebot_ob.reshape(3, 2), 
                                self.target.reshape(1, 2)), dtype=np.float32)

    # this is render function for enable GUI display
    def render(self, mode='human'):
        pass

    # for shut down and disconnect physical client/server
    def close(self):
        p.disconnect(self.client)