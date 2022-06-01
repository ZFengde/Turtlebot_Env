import gym
import numpy as np
import math
import pybullet as p
from turtlebot_env.resources.turtlebot import Turtlebot
from turtlebot_env.resources.plane import Plane
from turtlebot_env.resources.target import Target
from pybullet_utils import bullet_client

class TurtleBotEnv_Parallel(gym.Env):
    metadata = {'render.modes': ['human']}

    # this is for gym environment initialisation
    def __init__(self, batch_num = 5, use_gui=False):
        self.batch_num = batch_num
        self.clients = []
        if use_gui == True:
            for _ in range(self.batch_num):
                self.clients.append(bullet_client.BulletClient(connection_mode=p.GUI))
        else:
            for _ in range(self.batch_num):
                self.clients.append(bullet_client.BulletClient(connection_mode=p.DIRECT))

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
        self.turtlebot = []
        self.target = []
        self.prev_dist_to_target = []

    def step(self, actions):
        for ID in range(self.batch_num):
            self.step_per_env(self.clients[ID], actions[ID])

    def reset(self):
        for ID in range(self.batch_num):
            self.reset_per_env(self.clients[ID])

    def step_per_env(self, ID, action):
        # we need manually clip action input into (-1, 1) and then map it into desired velocity
        action = np.tanh(action)
        self.turtlebot.apply_action((action + 1) * 3.25)
        p.stepSimulation(self.clients[ID])
        turtlebot_ob = self.turtlebot.get_observation()

        # step reward setting, compute L2 distance firstly
        dist_to_target = math.sqrt(((turtlebot_ob[0] - self.target[0]) ** 2 +
                                  (turtlebot_ob[1] - self.target[1]) ** 2))

        # 1. foward reward, 2. time reward
        reward = 10 * (self.prev_dist_to_target[ID] - dist_to_target) - 0.01

        self.prev_dist_to_target[ID] = dist_to_target

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

    def reset_per_env(self, ID):
        p.resetSimulation(self.clients[ID])
        p.setGravity(0, 0, -9.8)
        # Reload the plane and car
        Plane(self.clients[ID])
        turtlebot = Turtlebot(self.clients[ID])

        # Set the target to a random target
        x = (self.np_random.uniform(1.3, 1.7) if self.np_random.randint(2) else
             self.np_random.uniform(-1.3, -1.7))
        y = (self.np_random.uniform(1.3, 1.7) if self.np_random.randint(2) else
             self.np_random.uniform(-1.3, -1.7))

        # self.target is the base position of the target
        self.target = np.array((x, y), dtype=float)
        
        self.done = False

        # Visual element of the target
        Target(self.client, self.target)

        # Get observation to return
        turtlebot_ob = turtlebot.get_observation()

        # this is for generating first prev_dist_to_target when initialising
        # for use in step function
        self.prev_dist_to_target.append(math.sqrt(((turtlebot_ob[0] - self.target[0]) ** 2 +
                                                     (turtlebot_ob[1] - self.target[1]) ** 2)))

        return np.concatenate((turtlebot_ob.reshape(3, 2), 
                                self.target.reshape(1, 2)), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def close(self):
        for ID in range(self.batch_num):
            p.disconnect(self.clients[ID])

