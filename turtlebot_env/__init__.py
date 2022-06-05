# this is for connecting entry point 

from gym.envs.registration import register

# fixed target at x, y = (1.5, 1.5)
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Simple'
)

# random target with distance related reward
register(
    id='Turtlebot-v1', 
    entry_point='turtlebot_env.envs:TurtleBotEnv'
)

# random target with sparse reward
register(
    id='Turtlebot-v2', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Sparse'
)

# parallel version environment, not yet finished
# TODO
register(
    id='Turtlebot-v3', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Parallel'
)

# random target with distance related reward, but speed up and break real features of the robot
# WORKING RIGHT NOW
register(
    id='Turtlebot-v4', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Speedup'
)