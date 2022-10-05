# this is for connecting entry point 

from gym.envs.registration import register

# navigation control, regular reward
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_0',
    max_episode_steps=1000
)

# navigation contrl, fuzzy reward
register(
    id='Turtlebot-v1',
    entry_point='turtlebot_env.envs:TurtleBotEnv_FuzzyReward_1',
    max_episode_steps=1000
)

# tracking control, regular reward
register(
    id='Turtlebot-v2',
    entry_point='turtlebot_env.envs:TurtleBotEnv_2',
    max_episode_steps=1500
)

# regular control, fuzzy reward
register(
    id='Turtlebot-v3',
    entry_point='turtlebot_env.envs:TurtleBotEnv_FuzzyReward_3',
    max_episode_steps=1500
)

# regular control, fuzzy reward
register(
    id='Turtlebot-v4',
    entry_point='turtlebot_env.envs:TurtleBotEnv_4',
    max_episode_steps=1500
)