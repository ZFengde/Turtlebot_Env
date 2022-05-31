# Turtlebot3 Pybullet Gym Environment

Clone the code, and then install the environment from root directory (where setup.py is):

```
pip install -e .
```

Then, in Python:

If GUI is using, then statement should be added when create envionrment, default is False

```
import gym 
import turtlebot_env
env = gym.make('Turtlebot-v0', use_gui=True) 
```

v0: Fixed goal
v1: Random goal, but dense reward
v2: Random goal, with sparse reward
