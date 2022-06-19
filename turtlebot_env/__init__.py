# this is for connecting entry point 

from gym.envs.registration import register

# fixed target at x, y = (1.3, 1.7)
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Fix_xy',
    max_episode_steps=1000
)

# # fixed target at x = (1.3, 1.7), y = +-(1.3, 1.7)
register(
    id='Turtlebot-v1', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Fix_x',
    max_episode_steps=1000
)