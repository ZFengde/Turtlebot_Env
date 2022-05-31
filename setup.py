# signifies this directory and whose sub contents are an installable package
# and is run when pip is used
# specify meta-data and dependencies

from setuptools import setup
setup(    
    name="turtlebot_env",
    version='0.0.1',
    install_requires=['gym', 'pybullet', 'numpy', 'matplotlib']
)