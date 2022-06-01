# this is for connecting entry point 

from gym.envs.registration import register

register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Simple'
)
register(
    id='Turtlebot-v1', 
    entry_point='turtlebot_env.envs:TurtleBotEnv'
)

register(
    id='Turtlebot-v2', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Sparse'
)

register(
    id='Turtlebot-v3', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Parallel'
)