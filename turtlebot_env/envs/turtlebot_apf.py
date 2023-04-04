# v4 --- obstacles avoidance environment with inherent artifical potential field implementation
import gym
import numpy as np
import math
import pybullet as p
from turtlebot_env.resources.turtlebot import Turtlebot
from turtlebot_env.resources.plane import Plane
from turtlebot_env.resources.target import Target
from turtlebot_env.resources.obstacle import Obstacle

class TurtleBotEnv_APF(gym.Env):
    metadata = {'render.modes': ['human']}

    # this is for gym environment initialisation
    def __init__(self, use_gui=False, obstacle_num=11):
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
        self.prev_dist_robot_obstalces = None

        self.radius_target = 0.05
        self.k_a = 5
        self.rho_target = 1.5
        self.k_b = self.rho_target * self.k_a

        self.radius_obstacle = 0.17
        self.rho_obstacle = 0.3
        self.k_o = 8

    # this is what happened in every single step
    def step(self, action):

        self.turtlebot.apply_action(action)
        p.stepSimulation()

        turtlebot_ob = self.turtlebot.get_observation() 
        obs = np.concatenate((turtlebot_ob, self.target, self.obstacle_bases.flatten()))

        pos = obs[:2]
        target = obs[6: 8]

        dist_to_target = np.linalg.norm(pos - target)
        dist_robot_obstalces = np.linalg.norm((pos - self.obstacle_bases), axis=1)
        
        reward = 0
        # terminate mode
        if (turtlebot_ob[0] >= 1.95 or turtlebot_ob[0] <= -1.95 or
            turtlebot_ob[1] >= 1.95 or turtlebot_ob[1] <= -1.95):
            self.done = True
            reward -= 20
        elif dist_to_target < 0.15:
            self.done = True
            reward += 70
            self.info['Success'] = True

        if not self.done:
            if min(dist_robot_obstalces) < 0.5:
            # penalty mode
                for i in range(len(dist_robot_obstalces)):
                    if dist_robot_obstalces[i] < 0.27:
                        reward -= 0.50
                        self.info['Collision'] = True
                    elif dist_robot_obstalces[i] < 0.5:
                        # reward += 50 * (self.prev_dist_robot_obstalces[i] - dist_robot_obstalces[i])
                        reward += 0
            # reward mode
            else:
                reward += 20 * (self.prev_dist_to_target - dist_to_target) - 0.01

        self.prev_dist_to_target = dist_to_target
        self.prev_dist_robot_obstalces = dist_robot_obstalces
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
        x = -1.7
        y = np.random.uniform(-1.7, 1.7)
        pos = np.array([x, y])
        self.turtlebot = Turtlebot(self.client, Pos=pos)

        # self.target is the base position of the target
        self.obstacle_bases = np.random.uniform(low=(-1.5, -1.5), high=(1.5, 1.5), size=(self.obstacle_num, 2))
        self.done = False

        x_target = 1.7
        y_target = np.random.uniform(-1.7, 1.7)
        # Visual element of the target
        self.target = np.array([x_target, y_target])
        Target(self.client, self.target)
        for i in range(len(self.obstacle_bases)):
            Obstacle(self.client, self.obstacle_bases[i])

        # Get observation to return
        turtlebot_ob = self.turtlebot.get_observation()

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
            cos_value = np.clip(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
            arccos = math.acos(cos_value)
            angle = np.degrees(arccos)
            return angle

    def generate_action(self, obs):
        pos = obs[:2]
        vel = obs[4: 6]
        target = obs[6: 8]
        
        e_target = target - pos
        dist_target = np.linalg.norm(pos - target) - self.radius_target -0.1

        # conical potential field of the target
        if dist_target <= self.rho_target:
            f_target = self.k_a * e_target
        else:
            f_target = self.k_b * e_target

        # potential field of the obstacles
        e_obstacle = self.obstacle_bases - pos
        dist_obstacle = np.linalg.norm((pos - self.obstacle_bases), axis=1) - self.radius_obstacle - 0.1

        f_obstacle = 0
        for i in range(len(dist_obstacle)):
            if dist_obstacle[i] < self.rho_obstacle:
                f_obstacle -= 1/(dist_obstacle[i] ** 2) * (1/dist_obstacle[i] - 1/self.rho_obstacle) * e_obstacle[i]
            else:
                pass
            
        f_total = f_target + f_obstacle

        theta_f = np.arctan2(f_total[1], f_total[0])
        theta = np.arctan2(vel[1], vel[0])
        # v = (f_total[1] * np.cos(theta) +  f_total[0] + np.sin(theta)) * 3
        v = 20
        w = (-theta_f + theta) * 50 # w positive, turn right, w negative turn left

        '''
        action space initialisation
        should be the rotation speed omiga here
        radius = 0.033, D = 0.22
        V = (wl+wr)/2, W = (wl - wr)/D
        '''
        wl = (2 * v + 0.22 * w)/2
        wr = (2 * v - 0.22 * w)/2

        action = np.array([wl, wr])

        return action