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
    WIND_UP_TIME = 3
    OUTWARD_DURATION = 3

    # Initialization.
    def __init__(self, node):
        # JOINT SPACE: Joint positions, kinematic chains, and rotation matrices
        self.q = np.radians(np.zeros((48, 1)))

        # LEFT SIDE
        # ARM: pelvis, stomach, abs, lowerChest, upperChest, leftInnerShoulder, leftShoulder, leftElbow, leftWrist
        self.q_left_arm = np.radians(np.zeros((16, 1)))
        self.q0_left_arm = np.radians(np.zeros((16, 1)))
        self.q_left_arm[12] = np.radians(-1.804)  # leftShoulder_rotz = -1.804
        self.chain_left = KinematicChain(node, 'Pelvis', 'LeftHand_f1', self.joints_by_chain("pelvis_to_left_arm"))
        self.R_left = Reye()

        # RIGHT SIDE
        # pelvis, stomach, abs, lowerChest, upperChest, rightInnerShoulder, rightShoulder, rightElbow, rightWrist
        self.q_right_arm = np.radians(np.zeros((16, 1)))
        self.q0_right_arm = np.radians(np.zeros((16, 1)))
        self.q_right_arm[12] = np.radians(1.826)  # rightShoulder_rotz = 1.826
        self.chain_right = KinematicChain(node, 'Pelvis', 'RightHand_f1', self.joints_by_chain("pelvis_to_right_arm"))
        self.R_right = Reye()


        # TASK SPACE
        self.error_left_arm = np.zeros((6,1))
        self.error_right_arm = np.zeros((6,1))
        self.error_arm = np.vstack((self.error_left_arm, self.error_right_arm))

        # # TASK 1
        # # Initial
        # self.pleft_arm  = np.array([0.53274, 0.19861, 0.43292]).reshape((3,1))
        # self.pright_arm = np.array([-0.37595, -0.56397, 0.43302]).reshape((3,1))

        # TASK 2
        # Initial
        self.pin_left_arm = np.array([0.51843, 0.064053, 0.43283]).reshape((3, 1))
        self.pin_right_arm = np.array([0.51569, -0.953098, 0.43278]).reshape((3, 1))
        # Final
        self.pout_left_arm = np.array([0.60774, 0.064053, 0.11712]).reshape((3, 1))
        self.pout_right_arm = np.array([0.60538, -0.053098, 0.1185]).reshape((3, 1))
        
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
    
    def fill_jac(self, partial_jac, chain):
        jac = np.zeros((6, 48))
        keep = self.joint_indicies_by_chain(chain)
        jac[:, keep] = partial_jac
        return jac

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        # return np.zeros((48, 1)).flatten().tolist(), np.zeros((48, 1)).flatten().tolist()
        
        # Hands come together from both sides to join
        if t < self.WIND_UP_TIME:
        #     pd = 
        #     vd = 

        #     Rd = Reye()
        #     wd = np.zeros((3,1))
            # pass    

        # Go outwards
        # elif self.WIND_UP_TIME < t < self.WIND_UP_TIME + self.OUTWARD_DURATION:
            w = -pi # Frequency
            t1 = (t-self.WIND_UP_TIME) % self.OUTWARD_DURATION

            # Move arm inwards
            sp = - cos(w * (t - self.WIND_UP_TIME))
            spdot = w * sin(w * (t - self.WIND_UP_TIME))

            # Use the path variables to compute the trajectory.
            pd_left = (0.5*(self.pout_left_arm+self.pin_left_arm) + 0.5*(self.pout_left_arm-self.pin_left_arm) * sp).reshape((3, 1))
            vd_left = (0, (0.5*(self.pout_left_arm[1]-self.pin_left_arm[1]) * spdot), 0).reshape((3, 1))

            pd_right = (0.5*(self.pout_right_arm+self.pin_right_arm) + 0.5*(self.pout_right_arm-self.pin_right_arm) * sp).reshape((3, 1))
            vd_right = (0.5*(self.pout_right_arm-self.pin_right_arm) * spdot).reshape((3, 1))
            
            Rd_left = Reye()
            wd_left = np.zeros((3, 1))
            Rd_right = Reye()
            wd_right = np.zeros((3, 1))
        
        qlast = self.q
        error = self.error_arm

        (ptip_left, R_left, Jv_left, Jw_left) = self.chain_left.fkin(qlast[self.joint_indicies_by_chain("pelvis_to_left_arm")])
        (ptip_right, R_right, Jv_right, Jw_right) = self.chain_right.fkin(qlast[self.joint_indicies_by_chain("pelvis_to_right_arm")])


        J_left = self.fill_jac(np.vstack((Jv_left, Jw_left)), "pelvis_to_left_arm")
        v_left = np.vstack((vd_left, wd_left))
        J_right = self.fill_jac(np.vstack((Jv_right, Jw_right)), "pelvis_to_right_arm")
        v_right = np.vstack((vd_right, wd_right))
        J = np.vstack((J_left, J_right))
        v = np.vstack((v_left, v_right))

        Jinv = np.transpose(J)@np.linalg.inv(J@np.transpose(J))
        qdot = Jinv @ (v + self.lam*error)
        q = qlast + dt*qdot

        error_left = np.vstack((ep(pd_left, ptip_left), eR(Rd_left, R_left)))
        error_right = np.vstack((ep(pd_right, ptip_right), eR(Rd_right, R_right)))

        self.error = np.vstack((error_left, error_right))
        self.q = q

        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist())

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
