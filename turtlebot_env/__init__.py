# this is for connecting entry point 

from gym.envs.registration import register

# fixed target at x, y = (1.3, 1.7)
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Fix_xy',
    max_episode_steps=1000
)

# target fixed at x = (1.3, 1.7), but random y = +-(1.3, 1.7)
register(
    id='Turtlebot-v1', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Fix_x',
    max_episode_steps=1000
)

# random target at x = +-(1.3, 1.7), y = +-(1.3, 1.7)
register(
    id='Turtlebot-v2', 
    entry_point='turtlebot_env.envs:TurtleBotEnv',
    max_episode_steps=1000
)