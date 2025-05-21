
import gymnasium as gym
import numpy as np
import turtlebot_env
from stable_baselines3 import TD3, SAC, PPO, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make('Humanoid-v4', csv_path = './output.csv') 
env = gym.make('Turtlebot-v3') 

# The noise objects for TD3                 
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_starts=3000)
# model = A2C("MlpPolicy", env, verbose=1, n_steps=2048)
model.learn(total_timesteps=1000000, log_interval=1)

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    scaled_action = model.policy.scale_action(action)
    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
    # We store the scaled action in the buffer
    buffer_action = scaled_action
    action = model.policy.unscale_action(scaled_action)
    obs, reward, terminated, truncated, info = env.step(action)
    if truncated:
        obs, _ = env.reset()