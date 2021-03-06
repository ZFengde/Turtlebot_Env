# v1 --- random x, y, fuzzy reward
import gym
import numpy as np
import math
import pybullet as p
from turtlebot_env.resources.turtlebot import Turtlebot
from turtlebot_env.resources.plane import Plane
from turtlebot_env.resources.target import Target
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class TurtleBotEnv_Fuzzy_2_Reward(gym.Env):
    metadata = {'render.modes': ['human']}

    # this is for gym environment initialisation
    def __init__(self, use_gui=False):
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
        self.target = None
        self.prev_dist_to_target = None

        # initialise self fuzzy inference system
        self._init_fuzzy_system()

    # this is what happened in every single step
    def step(self, action):
        self.turtlebot.apply_action((action + 1) * 3.25 * 5)
        p.stepSimulation()
        turtlebot_ob = self.turtlebot.get_observation()
        obs = np.concatenate((turtlebot_ob, self.target))

        # 1. foward reward, 2. time reward
        reward = self.fuzzy_reward_calculation(obs)

        # 3. Done by running off boundaries penalty
        if (turtlebot_ob[0] >= 2 or turtlebot_ob[0] <= -2 or
                turtlebot_ob[1] >= 2 or turtlebot_ob[1] <= -2):
            self.done = True
            reward = -10

        # 4. Done by reaching target reward
        elif self.dist_to_target < 0.15:
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
        # Reload the plane and car
        Plane(self.client)
        self.turtlebot = Turtlebot(self.client)

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
        turtlebot_ob = self.turtlebot.get_observation()

        # this is for generating first prev_dist_to_target when initialising
        # for use in step function
        self.prev_dist_to_target = math.sqrt(((turtlebot_ob[0] - self.target[0]) ** 2 +
                                           (turtlebot_ob[1] - self.target[1]) ** 2))
        obs = np.concatenate((turtlebot_ob, self.target))
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

    def fuzzy_reward_calculation(self, obs):
        pos = obs[:2]
        ori = obs[2: 4]
        vel = obs[4: 6]
        target = obs[6:]

        alpha = target - pos
        beta = vel + ori
        error_angle = self.angle(alpha, beta)
        self.dist_to_target = np.linalg.norm(pos - target)
        delta_distance = self.prev_dist_to_target - self.dist_to_target
        self.prev_dist_to_target = self.dist_to_target

        self.reward_system.input['e_d'] = delta_distance * 1000
        self.reward_system.input['e_a'] = error_angle
        self.reward_system.compute()

        # -0.01 is time elapse negative reward
        reward = self.reward_system.output['reward'] - 0.01
        return reward

    def _init_fuzzy_system(self):
        e_d = ctrl.Antecedent(np.arange(-4.5, 4.6, 0.1), 'e_d')
        e_a = ctrl.Antecedent(np.arange(0, 181, 1), 'e_a')
        reward = ctrl.Consequent(np.arange(-0.1, 0.11, 0.01), 'reward')

        e_d['small'] = fuzz.gaussmf(e_d.universe, -4.5, 1.5)
        e_d['medium'] = fuzz.gaussmf(e_d.universe, 0, 1.5)
        e_d['large'] = fuzz.gaussmf(e_d.universe, 4.5, 1.5)

        e_a['small'] = fuzz.gaussmf(e_a.universe, 0, 30)
        e_a['medium'] = fuzz.gaussmf(e_a.universe, 90, 30)
        e_a['large'] = fuzz.gaussmf(e_a.universe, 180, 30)

        reward['small'] = fuzz.gaussmf(reward.universe, -0.1, 0.033)
        reward['medium'] = fuzz.gaussmf(reward.universe, 0, 0.033)
        reward['large'] = fuzz.gaussmf(reward.universe, 0.1, 0.033)

        rule1 = ctrl.Rule(antecedent=((e_d['large'] & e_a['small']) |
                                    (e_d['large'] & e_a['medium']) |
                                    (e_d['medium'] & e_a['small'])),
                        consequent=reward['large'])

        rule2 = ctrl.Rule(antecedent=((e_d['large'] & e_a['large']) |
                                    (e_d['medium'] & e_a['medium']) |
                                    (e_d['small'] & e_a['small'])),
                        consequent=reward['medium'])

        rule3 = ctrl.Rule(antecedent=((e_d['medium'] & e_a['large']) |
                                    (e_d['small'] & e_a['medium']) |
                                    (e_d['small'] & e_a['large'])),
                        consequent=reward['small'])

        reward_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.reward_system = ctrl.ControlSystemSimulation(reward_ctrl)
