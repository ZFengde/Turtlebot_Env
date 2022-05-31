import pybullet as p
import os
import math
import numpy as np

class turtlebot:
    def __init__(self, client):
        # firstly initialisation with given client ID
        self.client = client
        # create turtlebot with given urdf file
        f_name = os.path.join(os.path.dirname(__file__), 'turtlebot.urdf')
        self.turtlebot = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.03],
                              physicsClientId=client)

        # 0 - wheel_left_joint
        # 1 - wheel_right_joint
        self.drive_joints = [0, 1]

    def get_ids(self):
        # get body uniqued id and physics client/server id
        return self.turtlebot, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional, unpack action 
        left_wheel, right_wheel = action.squeeze()

        p.setJointMotorControlArray(
            bodyUniqueId=self.turtlebot,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_wheel, right_wheel],
            forces=[1.2] * 2,
            physicsClientId=self.client)

    # define how get observation can obtain
    def get_observation(self):
        # Get the position and orientation of the turtlebot in the simulation
        # where pos is xyz, and orientation xyzw quaternion direction
        pos, ang = p.getBasePositionAndOrientation(self.turtlebot, self.client)
        ang = p.getEulerFromQuaternion(ang)

        # here we obtain the horizontal derection
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]

        '''
        Get the velocity of the turtlebot
        so in pybullet we actually have method for directly obtain velocity 
        in the form of xyz and wx, wy, wz in Cartesian worldspace coordinates
        '''
        vel = p.getBaseVelocity(self.turtlebot, self.client)[0][0:2]
        # Concatenate/pack position, orientation, velocity
        observation = np.array(pos + ori + vel)

        return observation








