# this is for connecting entry point 

from gym.envs.registration import register

# navigation control, regular reward
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Navi',
    max_episode_steps=1000
)

# regular control, fuzzy reward
register(
    id='Turtlebot-v1',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Constrained_Easy',
    max_episode_steps=1500
)

# regular control, fuzzy reward
register(
    id='Turtlebot-v2',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Constrained',
    max_episode_steps=1500
)

