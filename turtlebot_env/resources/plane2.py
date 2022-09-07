import pybullet as p
import os

# create plane class, with given urdf file and clinet ID
# by specifying base position
class Plane2:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'plane2.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[0, 0, 0],
                   physicsClientId=client)

