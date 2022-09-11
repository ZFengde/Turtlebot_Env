# v2 --- regular reward, tracking control
import gym
import numpy as np
import math
import pybullet as p
from turtlebot_env.resources.turtlebot import Turtlebot
from turtlebot_env.resources.plane2 import Plane2

class TurtleBotEnv_2(gym.Env):
    metadata = {'render.modes': ['human']}

    # this is for gym environment initialisation
    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        if self.use_gui:
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
        self.prev_dist_to_target = None

    # this is what happened in every single step
    def step(self, action):
        self.turtlebot.apply_action((action + 1) * 3.25 * 5)
        self.target.apply_action(np.random.rand(2) * 3.25 * 5) # half velocity

        p.stepSimulation()

        turtlebot_ob = self.turtlebot.get_observation()
        target_pos = self.target.get_observation()[:2]

        obs = np.concatenate((turtlebot_ob, target_pos))

        pos = obs[:2]
        ori = obs[2: 4]
        vel = obs[4: 6]
        target = obs[6:]
        alpha = target - pos
        beta = vel + ori
        error_angle = self.angle(alpha, beta)

        # step reward setting, compute L2 distance firstly
        dist_to_target = np.linalg.norm(pos - target)

        # 1. foward reward, 2. time reward
        reward = 7 * (self.prev_dist_to_target - dist_to_target) - 3.5e-4 * (error_angle - 90) - 0.05
    
        self.prev_dist_to_target = dist_to_target

        # 3. Done by running off boundaries penalty
        if dist_to_target > 6.5:
            self.done = True
            reward = -10

        # 4. Done by reaching target reward
        elif dist_to_target < 0.2:
            self.done = True
            reward = 50
            self.info['Success'] = 'Yes'

        return obs, reward, self.done, self.info

    # this is for generating random seeds for training
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # this is reset function for initialising each episode
    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        self.done = False

        # Set the target to a random target
        x_target = self.np_random.uniform(-1.5, -1.7)
        y_target = (self.np_random.uniform(1.5, 1.7) if self.np_random.randint(2) else
             self.np_random.uniform(-1.5, -1.7))

        Plane2(self.client)
        self.target = Turtlebot(self.client, Pos = [x_target, y_target, 0.03])
        self.turtlebot = Turtlebot(self.client, Pos=[-2.5, -2.5, 0.03])

        turtlebot_ob = self.turtlebot.get_observation()
        target_pos = self.target.get_observation()[:2]

        self.prev_dist_to_target = math.sqrt(((turtlebot_ob[0] - target_pos[0]) ** 2 +
                                           (turtlebot_ob[1] - target_pos[1]) ** 2))
        obs = np.concatenate((turtlebot_ob, target_pos))
        self.info = {'Success': 'No'}
        
        return obs

    # this is render function for enable GUI display
    def render(self, mode='human'):
        pass

    # for shut down and disconnect physical client/server
    def close(self):
        p.disconnect(self.client)

    def angle(self, v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        else:
            vector_dot_product = np.dot(v1, v2)
            arccos = math.acos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(arccos)
            return angle
