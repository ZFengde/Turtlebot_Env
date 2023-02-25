# v3 --- obstacles avoidance environment with reward function
import gym
import numpy as np
import math
import pybullet as p
from turtlebot_env.resources.turtlebot import Turtlebot
from turtlebot_env.resources.plane import Plane
from turtlebot_env.resources.target import Target
from turtlebot_env.resources.obstacle import Obstacle

class TurtleBotEnv_Constrained_Reward(gym.Env):
    metadata = {'render.modes': ['human']}

    # this is for gym environment initialisation
    def __init__(self, use_gui=False, obstacle_num=5):
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
        
        # this is for random initialisation, could be replaced by another method
        self.np_random, _ = gym.utils.seeding.np_random()

        # placeholders
        self.turtlebot = None
        self.target = None
        self.prev_dist_to_target = None
        self.prev_dist_robot_obstalces = None

        # self.obstacle_bases = np.random.uniform(low=(-0.8, -0.8), high=(0.8, 0.8), size=(self.obstacle_num, 2))

    # this is what happened in every single step
    def step(self, action):
          
        # TODO, different action taken, but not too different
        # # originally, action = [wl, wr]
        # # TEST, here we try if action is (v, w)
        # diameter = 0.22 
        # wl = (2 * action[0] + action[1] * diameter)/2
        # wr = (2 * action[0] - action[1] * diameter)/2
        # action = (np.array([wl, wr])) * 3.25 * 10
        # # maximum [32.5, 32.5]
        # self.turtlebot.apply_action(action)

        self.turtlebot.apply_action((action + 1) * 3.25 * 5)

        p.stepSimulation()
        turtlebot_ob = self.turtlebot.get_observation() 
        obs = np.concatenate((turtlebot_ob, self.target, self.obstacle_bases.flatten()))

        pos = obs[:2]
        ori = obs[2: 4]
        vel = obs[4: 6]
        target = obs[6: 8]
        alpha = target - pos
        beta = vel + ori
        error_angle = self.angle(alpha, beta)

        # step reward setting, compute L2 distance firstly
        # dist_to_obstacles = 
        dist_to_target = np.linalg.norm(pos - target)

        # 1. foward reward, 2. time reward
        reward = 20 * (self.prev_dist_to_target - dist_to_target) / (dist_to_target + 1e-7) - 1e-4 * (error_angle - 90) - 0.01
        self.prev_dist_to_target = dist_to_target
        
        # 2. Done by running off boundaries penalty
        if (turtlebot_ob[0] >= 1.95 or turtlebot_ob[0] <= -1.95 or
                turtlebot_ob[1] >= 1.95 or turtlebot_ob[1] <= -1.95):
            self.done = True
            reward = -10

        # 3. Done by reaching target reward
        elif dist_to_target < 0.15:
            self.done = True
            reward = 50
            self.info['Success'] = 'Yes'

        # 4. Obstacles guiding reward and cost
        self.info['cost'] = 0
        dist_robot_obstalces = np.linalg.norm((pos - self.obstacle_bases), axis=1)
        for i in range(len(dist_robot_obstalces)):
            if dist_robot_obstalces[i] < 0.3:
                self.info['cost'] += 0.15
            elif dist_robot_obstalces[i] < 0.5:
                reward -= 10 * (dist_robot_obstalces[i] - self.prev_dist_robot_obstalces[i]) / (dist_robot_obstalces[i] + 1e-7)

        self.prev_dist_robot_obstalces = dist_robot_obstalces
        # obs: robot [: 6], target [6: 8], obstacles [8: ]
        return obs, reward, self.done, self.info

    # this is for generating random seeds for training
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # this is reset function for initialising each episode
    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        # Reload the plane and car
        Plane(self.client)
        x = -1.4
        y = np.random.uniform(-1.5, 1.5)
        pos = np.array([x, y])
        self.turtlebot = Turtlebot(self.client, Pos=pos)

        # self.target is the base position of the target
        self.obstacle_bases = np.random.uniform(low=(-0.8, -0.8), high=(0.8, 0.8), size=(self.obstacle_num, 2))
        self.done = False

        x_target = np.random.uniform(1.3, 1.7)
        y_target = np.random.uniform(-1.7, 1.7)
        # Visual element of the target
        self.target = np.array([x_target, y_target])
        Target(self.client, self.target)
        for i in range(len(self.obstacle_bases)):
            Obstacle(self.client, self.obstacle_bases[i])

        # Get observation to return
        turtlebot_ob = self.turtlebot.get_observation()

        # this is for generating first prev_dist_to_target when initialising
        # for use in step function
        self.prev_dist_to_target = np.linalg.norm(pos - self.target)
        self.prev_dist_robot_obstalces = np.linalg.norm((pos - self.obstacle_bases), axis=1)

        obs = np.concatenate((turtlebot_ob, self.target, self.obstacle_bases.flatten()))
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
