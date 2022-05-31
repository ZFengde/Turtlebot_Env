import pybullet as p
import os

# create target class, with given urdf file and clinet ID
# by specifying base position
class obstacle:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'obstacle.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0],
                   physicsClientId=client)

