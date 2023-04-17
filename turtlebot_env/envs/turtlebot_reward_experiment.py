# v3 --- obstacles avoidance environment with reward function
import gym
import numpy as np
import math
import pybullet as p
from turtlebot_env.resources.turtlebot import Turtlebot
from turtlebot_env.resources.plane import Plane
from turtlebot_env.resources.target import Target
from turtlebot_env.resources.obstacle import Obstacle

class TurtleBotEnv_Reward_Exp(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, use_gui=False, obstacle_num=7, indicator=1):
        self.use_gui = use_gui
        self.obstacle_num = obstacle_num
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
        
         # should >= 1
        low = np.concatenate((np.array([-5, -5, -1, -1, -3, -3, -5, -5]), np.ones(self.obstacle_num) * -5, np.ones(self.obstacle_num) * -5), dtype=np.float32)
        high = np.concatenate((np.array([5, 5, 1, 1, 3, 3, 5, 5]), np.ones(self.obstacle_num) * 5, np.ones(self.obstacle_num) * 5), dtype=np.float32)
        self.observation_space = gym.spaces.box.Box(low=low, high=high)
        
        # placeholders
        self.turtlebot = None
        self.target = None
        self.prev_dist_to_target = None
        self.prev_dist_robot_obstalces = None
        self.init_hyperparameters(indicator)

    def init_hyperparameters(self, indicator):
        if indicator == 1:
            self.c_e_target, self.c_e_obstacles = 20, -40
            self.reach_target, self.collision = 50, -0.15
            self.out, self.time_penalty = -10, -0.01

        elif indicator == 2:
            self.c_e_target, self.c_e_obstacles = 20, -40
            self.reach_target, self.collision = 50, -0.2
            self.out, self.time_penalty = -10, -0.01

        elif indicator == 3:
            self.c_e_target, self.c_e_obstacles = 20, -40
            self.reach_target, self.collision = 50, -0.25
            self.out, self.time_penalty = -10, -0.01

        elif indicator == 4:
            self.c_e_target, self.c_e_obstacles = 20, -40
            self.reach_target, self.collision = 50, -0.3
            self.out, self.time_penalty = -10, -0.01

        elif indicator == 5:
            self.c_e_target, self.c_e_obstacles = 20, -40
            self.reach_target, self.collision = 50, -0.35
            self.out, self.time_penalty = -10, -0.01


    def step(self, action):
        self.turtlebot.apply_action((action + 1) * 3.25 * 5)

        p.stepSimulation()
        turtlebot_ob = self.turtlebot.get_observation() 
        obs = np.concatenate((turtlebot_ob, self.target, self.obstacle_bases.flatten()))

        pos = obs[:2]
        target = obs[6: 8]

        dist_to_target = np.linalg.norm(pos - target)
        dist_robot_obstalces = np.linalg.norm((pos - self.obstacle_bases), axis=1)

        reward = self.c_e_target * (self.prev_dist_to_target - dist_to_target) + self.time_penalty
        
        if (turtlebot_ob[0] >= 2.2 or turtlebot_ob[0] <= -2.2 or
                turtlebot_ob[1] >= 2.2 or turtlebot_ob[1] <= -2.2):
            self.done = True
            reward = self.out

        elif dist_to_target < 0.15:
            self.done = True
            reward = self.reach_target
            self.info['Success'] = True

        for i in range(len(dist_robot_obstalces)):
            if dist_robot_obstalces[i] < 0.27:
                reward += self.collision
                self.info['Collision'] = True
            if dist_robot_obstalces[i] < 0.6:
                reward += self.c_e_obstacles * (self.prev_dist_robot_obstalces[i] - dist_robot_obstalces[i])

        self.prev_dist_to_target = dist_to_target
        self.prev_dist_robot_obstalces = dist_robot_obstalces
        return obs, reward, self.done, self.info
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        Plane(self.client)
        x = -1.7
        y = np.random.uniform(-1.7, 1.7)
        pos = np.array([x, y])
        self.turtlebot = Turtlebot(self.client, Pos=pos)

        x_target = 1.7
        y_target = np.random.uniform(-1.7, 1.7)

        self.target = np.array([x_target, y_target])

        self.obstacle_bases = np.random.uniform(low=(-0.8, -0.8), high=(0.8, 0.8), size=(self.obstacle_num, 2))

        self.done = False
        Target(self.client, self.target)
        for i in range(len(self.obstacle_bases)):
            Obstacle(self.client, self.obstacle_bases[i])

        turtlebot_ob = self.turtlebot.get_observation()

        self.prev_dist_to_target = np.linalg.norm(pos - self.target)
        self.prev_dist_robot_obstalces = np.linalg.norm((pos - self.obstacle_bases), axis=1)

        obs = np.concatenate((turtlebot_ob, self.target, self.obstacle_bases.flatten()))
        self.info = {'Success': False, 'Collision': False}

        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)

    def angle(self, v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        else:
            vector_dot_product = np.dot(v1, v2)
            cos_value = np.clip(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
            arccos = math.acos(cos_value)
            angle = np.degrees(arccos)
            return angle
