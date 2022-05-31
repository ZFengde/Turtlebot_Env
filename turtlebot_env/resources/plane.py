import pybullet as p
import os

# create plane class, with given urdf file and clinet ID
# by specifying base position
class plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'plane.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[0, 0, 0],
                   physicsClientId=client)

