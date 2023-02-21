# this is for connecting entry point 

from gym.envs.registration import register

# navigation control, regular reward
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Navi',
    max_episode_steps=1000
)

# obstacle aviodance, with constrained problem setting
register(
    id='Turtlebot-v1',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Constrained',
    max_episode_steps=2000
)

# obstacle aviodance, with unconstrained problem setting
register(
    id='Turtlebot-v2',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Constrained_Reward',
    max_episode_steps=2000
)

# obstacle aviodance, with unconstrained problem setting
register(
    id='Turtlebot-v3',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Constrained_Test',
    max_episode_steps=2000
)

