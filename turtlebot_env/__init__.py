# this is for connecting entry point 

from gym.envs.registration import register

# navigation control, regular reward
register(
    id='Turtlebot-v0', 
    entry_point='turtlebot_env.envs:TurtleBotEnv_Navi',
    max_episode_steps=1000
)

# obstacle aviodance, non-terminated collision constraint setting
register(
    id='Turtlebot-v1',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Constrained',
    max_episode_steps=2000
)

# obstacle aviodance, non-terminated collision pure reward
register(
    id='Turtlebot-v2',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Reward_Nonterminal',
    max_episode_steps=2000
)

# obstacle aviodance, experimental setting, terminated collision pure reward
register(
    id='Turtlebot-v3',
    entry_point='turtlebot_env.envs:TurtleBotEnv_Reward_Test',
    max_episode_steps=2000
)

# obstacle environment based on artifical potential field method
register(
    id='Turtlebot-v4',
    entry_point='turtlebot_env.envs:TurtleBotEnv_APF',
    max_episode_steps=2000
)

