'''FinalProjectPreliminaryTesting

   This is the preliminary testing for the controls of HumanSubject06

   We will attempt to costrain the two hands together as they move forward and backward
   while the entire robot spins

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState
'''

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from figure_skating_robot.GeneratorNode      import GeneratorNode
from figure_skating_robot.TransformHelpers   import *
from figure_skating_robot.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from figure_skating_robot.KinematicChain     import KinematicChain

#
#   Trajectory Class
#
class Trajectory():
    WIND_UP_TIME = np.pi

    # Initialization.
    def __init__(self, node):
        # JOINT SPACE: Joint positions, kinematic chains, and rotation matrices
        self.q = np.radians(np.zeros((48, 1)))

        # LEFT SIDE
        # pelvis, stomach, abs, lowerChest, upperChest, leftInnerShoulder, leftShoulder, leftElbow, leftWrist
        self.q_left = np.radians(np.zeros((16, 1)))
        self.q0_left = np.radians(np.zeros((16, 1)))
        self.q_left[12] = np.radians(-1.550)  # leftShoulder_rotz = -1.550
        self.chain_left = KinematicChain(node, 'Pelvis', 'LeftHand_f1', self.joints_by_chain("pelvis_to_left_arm"))
        self.R_left = Reye()

        # RIGHT SIDE
        # pelvis, stomach, abs, lowerChest, upperChest, rightInnerShoulder, rightShoulder, rightElbow, rightWrist
        self.q_right = np.radians(np.zeros((16, 1)))
        self.q0_right = np.radians(np.zeros((16, 1)))
        self.q_right[12] = np.radians(-0.785)  # rightShoulder_rotz = -0.785
        self.chain_right = KinematicChain(node, 'Pelvis', 'RightHand_f1', self.joints_by_chain("pelvis_to_right_arm"))
        self.R_right = Reye()

        # TASK SPACE
        self.pleft  = np.array([0.53274, 0.19861, 0.43292]).reshape((3,1))
        self.error_left = np.zeros((6,1))
        self.pright = np.array([-0.37595, -0.56397, 0.43302]).reshape((3,1))
        self.error_right = np.zeros((6,1))
        
        self.lam = 20

    # Declare the joint names
    def jointnames(self):
        joints = [
            'stomach_rotx', 'stomach_roty', 
            'abs_rotx', 'abs_roty', 
            'lowerChest_rotx', 'lowerChest_roty', 
            'upperChest_rotx', 'upperChest_roty', 'upperChest_rotz', 
            'neck_rotx', 'neck_roty', 'neck_rotz', 
            'head_rotx', 'head_roty', 
            
            'rightInnerShoulder_rotx', 'rightShoulder_rotx', 'rightShoulder_roty', 'rightShoulder_rotz', 'rightElbow_roty', 'rightElbow_rotz', 'rightWrist_rotx', 'rightWrist_rotz', 'rightHip_rotx', 'rightHip_roty', 'rightHip_rotz', 'rightKnee_roty', 'rightKnee_rotz', 'rightAnkle_rotx', 'rightAnkle_roty', 'rightAnkle_rotz', 'rightBallFoot_roty', 
            
            'leftInnerShoulder_rotx', 'leftShoulder_rotx', 'leftShoulder_roty', 'leftShoulder_rotz', 'leftElbow_roty', 'leftElbow_rotz', 'leftWrist_rotx', 'leftWrist_rotz', 'leftHip_rotx', 'leftHip_roty', 'leftHip_rotz', 'leftKnee_roty', 'leftKnee_rotz', 'leftAnkle_rotx', 'leftAnkle_roty', 'leftAnkle_rotz', 'leftBallFoot_roty']
        return joints

    def joint_indicies_by_chain(self, chain):
        # Return a list of joint names based the kinematic chain requested (from URDF)
        if (chain == "pelvis_to_left_arm"):
            keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 33, 34, 35, 36, 37]
        elif (chain == "pelvis_to_right_arm"):
            keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20]
        else:
            ValueError("Please provide a valid kinematic chain joint sequence")
        return keep

    def mask(self, keep):
        mask = np.zeros(48)
        for idx in keep:
            mask[idx] = 1
        return np.array(mask)
    
    def joints_by_chain(self, chain):
        all_joints = np.array(self.jointnames())
        mask = self.mask(self.joint_indicies_by_chain(chain))
        chain_joints = np.array(all_joints)[mask.astype(bool)]
        return chain_joints   

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        return np.zeros((48, 1)).flatten().tolist(), np.zeros((48, 1)).flatten().tolist()
        
        # # Hands come together from both sides to join
        # if t < self.WIND_UP_TIME:
        #     pd = 
        #     vd = 

        #     Rd = Reye()
        #     wd = np.zeros((3,1))

        # # Go outwards
        # # elif:
        # #     pass
        # # Go back in

        # else:
        #     return None
        #     g = cos(2*pi/5 * (t-3.0)) #2pi/5 because we want entire movement to take 5s
        #     gdot = -2*pi/5 * sin(2*pi/5 * (t-3.0))

        #     t1 = (t-3) % 5.0
        #     if t1 < 2.5:
        #         #We want the starting orientation to be like that from g = 0 and end at g = 1
        #         #If we make it from -1 to 1 (like in 4), then the rotation path will be symmetric.
        #         #We dont want this since the right and left orientation arent a mirror of each other.
        #         (gR, gRdot) = goto(t1,     2.5, 0.0, 1.0)
        #     else:
        #         (gR, gRdot) = goto(t1-2.5, 2.5, 1.0, 0.0)

        #     pd = np.array([-0.3 * g,    0.50,0.9 - 0.75*g**2]).reshape((3,1))
        #     vd = np.array([-0.3 * gdot, 0, -2*0.75*g*gdot]).reshape((3,1))

        #     Rd = Roty(-pi/2 * gR) @ Rotz(pi/2 * gR)
        #     #Rd = Rote((ey()), -pi/2 * gR)
        #     wd = (ey()-ez())*(-pi/2 * gRdot)
        
        # qlast = self.q
        # error = self.error

        # # J = np.zeros((6, 48))
        # # v = np.zeros((6, 3))

        # (plast, R, Jv, Jw) = self.chain_left.fkin(qlast)

        # J = np.vstack((Jv, Jw))
        # v = np.vstack((vd, wd))

        # qdot = np.linalg.inv(J) @ (v + self.lam*error)
        # q = qlast + dt*qdot

        # self.error = np.vstack((ep(pd, plast), eR(Rd, R)))
        # self.q = q

        # # Return the position and velocity as python lists.
        # return (q.flatten().tolist(), qdot.flatten().tolist())

#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
