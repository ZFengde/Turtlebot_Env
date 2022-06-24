# this is for connecting entry point 

from gym.envs.registration import register

# random target at x = +-(1.3, 1.7), y = +-(1.3, 1.7)
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv',
    max_episode_steps=1000
)

# random target at x = +-(1.3, 1.7), y = +-(1.3, 1.7), but fuzzy reward
register(
    id='Turtlebot-v1',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Fuzzy_Reward',
    max_episode_steps=1000
)