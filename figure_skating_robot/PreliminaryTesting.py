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
    UP_DURATION = 3

    # Initialization.
    def __init__(self, node):
        # JOINT SPACE: Joint positions, kinematic chains, and rotation matrices
        self.q = np.radians(np.zeros((48, 1)))

        # LEFT SIDE
        # ARM: pelvis, stomach, abs, lowerChest, upperChest, leftInnerShoulder, leftShoulder, leftElbow, leftWrist
        self.q[34] = -1.804  # leftShoulder_rotz = -1.804
        self.chain_left_arm = KinematicChain(node, 'Pelvis', 'LeftHand', self.joints_by_chain("pelvis_to_left_arm"))
        self.chain_left_foot = KinematicChain(node, 'Pelvis', 'LeftFoot', self.joints_by_chain("pelvis_to_left_foot"))

        # RIGHT SIDE
        # pelvis, stomach, abs, lowerChest, upperChest, rightInnerShoulder, rightShoulder, rightElbow, rightWrist
        self.q[17] = -0.785
        self.chain_right_arm = KinematicChain(node, 'Pelvis', 'RightHand', self.joints_by_chain("pelvis_to_right_arm"))
        self.chain_right_foot = KinematicChain(node, 'Pelvis', 'RightFoot', self.joints_by_chain("pelvis_to_right_foot"))

        # TASK SPACE
        self.error_arm_prim = np.zeros((6,1))
        self.error_arm_sec = np.zeros((6,1))
        self.error_arm = np.vstack((self.error_arm_prim, self.error_arm_sec))
        
        self.error_left_foot = np.zeros((6,1))
        self.error_right_foot = np.zeros((6,1))
        self.error_legs = np.vstack((self.error_left_foot, self.error_right_foot))
        
        self.error = np.vstack((self.error_arm, self.error_legs))

        # TASK 1
        # Initial
        self.pstart_left_arm = self.chain_left_arm.fkin(self.q[self.joint_indicies_by_chain("pelvis_to_left_arm")])[0]
        self.pstart_right_arm = self.chain_right_arm.fkin(self.q[self.joint_indicies_by_chain("pelvis_to_right_arm")])[0]
        self.pstart_right_foot = np.array([-0.449653, -0.493177, -0.71984]).reshape((3,1))


        # TASK 2
        # Initial (Same as TASK 1 FINAL)
        self.pin_left_arm = np.array([0.51844, 0.0642861, 0.43287]).reshape((3, 1))
        self.pin_right_arm = np.array([0.51561, -0.0529214, 0.43287]).reshape((3, 1))
        self.prim_init = (self.pin_left_arm + self.pin_right_arm)/2
        left_orientation = np.array([0, 0, -0.784569])
        right_orientation = np.array([0, 0, 0.783293])
        self.Rprim_left = Rotz(left_orientation[2])
        self.Rprim_right = Rotz(right_orientation[2])
        self.Rprim_init = Rmid(self.Rprim_left, self.Rprim_right)
        self.Rsec_init = np.transpose(self.Rprim_left)@self.Rprim_right

        # Final
        # self.pout_left_arm = np.array([0.60774, 0.064053, 0.11712]).reshape((3, 1))
        # self.pout_right_arm = np.array([0.60538, -0.053098, 0.1185]).reshape((3, 1))
        self.pout_left_arm = np.array([[0.447716, -0.437091, 0]]).reshape((3, 1))
        self.pout_right_arm = np.array([0.4124, -0.110882, 0]).reshape((3, 1))
        self.prim_final = (self.pout_left_arm + self.pout_right_arm)/2
        self.Rprim_final = self.Rprim_init
        self.Rsec_final = self.Rsec_init
        

        self.pstart2_left_foot  = np.array([0.00024073, 0.0983878, -0.935553]).reshape((3,1))
        self.pstart2_right_foot = np.array([0.450104, -0.098388, -0.820714]).reshape((3,1))

        # TASK 3
        # Initial
        self.pstart3_left_foot  = np.array([0.00024073, 0.0983878, -0.4]).reshape((3,1))
        self.pstart3_right_foot = np.array([0.938602, -0.098388, 0.00730084]).reshape((3,1))
        
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
            keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 33, 34, 35, 36, 37, 38]
        elif (chain == "pelvis_to_right_arm"):
            keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21]
        elif (chain == "pelvis_to_right_foot"):
            keep = [22, 23, 24, 25, 26, 27, 28, 29]
        elif (chain == "pelvis_to_left_foot"):
            keep = [39, 40, 41, 42, 43, 44, 45, 46]
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
        # Hands come together from both sides to join
        if (t < self.WIND_UP_TIME):

            self.q[17] = -.785+(1.826 + 0.785)/3*t # Right Arm Wind Up

            if t > self.WIND_UP_TIME/2:
                self.q[22] = -.50 + (0 + 0.50)/1.5*(t-1.5) # Right Leg Rotx Wind Up
            else:
                self.q[22] = -0.5
            
            self.q[23] = 0.50 + (-0.50 - 0.50)/3*t # Right Leg Roty Wind Up

            return self.q.flatten().tolist(), np.zeros((48, 1)).flatten().tolist() # HERE
            
        # Go outwards
        elif self.WIND_UP_TIME < t < self.WIND_UP_TIME + self.OUTWARD_DURATION + self.UP_DURATION:
            w = -pi/3 # Frequency

            # Move arm outwards
            sp = - cos(w * (t - self.WIND_UP_TIME))
            spdot = w * sin(w * (t - self.WIND_UP_TIME))

            # PRIMARY
            pd_1 = (0.5*(self.prim_final+self.prim_init) + 0.5*(self.prim_final-self.prim_init) * sp)
            vd_1 = (0.5*(self.prim_final-self.prim_init) * spdot)
            Rd_1 = Rinter(self.Rprim_init, self.Rprim_final, sp)
            wd_1 = winter(self.Rprim_init, self.Rprim_final, spdot)

            # SECONDARY
            p2_init = self.pin_left_arm - self.pin_right_arm
            T = 5.0
            pd_2, vd_2 = goto(t, T, p2_init, np.zeros((3, 1)))
            Rd_2 = Rinter(self.Rsec_init, self.Rsec_final, sp)
            wd_2 = winter(self.Rsec_init, self.Rsec_final, spdot)

            #Path for Right Leg
            pd_rightfoot = (0.5*(self.pstart3_right_foot + self.pstart2_right_foot) + 0.5*(self.pstart3_right_foot - self.pstart2_right_foot) * sp)
            vd_rightfoot = (0.5*(self.pstart3_right_foot - self.pstart2_right_foot) * spdot)
            Rd_rightfoot = Reye()
            wd_rightfoot = np.zeros((3, 1))

            #Path for Left Leg
            pd_leftfoot = (0.5*(self.pstart3_left_foot + self.pstart2_left_foot) + 0.5*(self.pstart3_left_foot - self.pstart2_left_foot) * sp)
            vd_leftfoot = (0.5*(self.pstart3_left_foot - self.pstart2_left_foot) * spdot)
            Rd_leftfoot = Reye()
            wd_leftfoot = np.zeros((3, 1))

        else: 
            # return self.q.flatten().tolist(), np.zeros((48, 1)).flatten().tolist()
            pass
        
        qlast = self.q

        #FKIN ON KINEMATIC CHAIN FPR LEFT ARM
        (ptip_left, R_left, Jv_left, Jw_left) = self.chain_left_arm.fkin(qlast[self.joint_indicies_by_chain("pelvis_to_left_arm")])

        #FKIN ON KINEMATIC CHAIN FPR RIGHT ARM
        (ptip_right, R_right, Jv_right, Jw_right) = self.chain_right_arm.fkin(qlast[self.joint_indicies_by_chain("pelvis_to_right_arm")])

        #FKIN ON KINEMATIC CHAIN FOR LEFT FOOT
        (ptip_leftfoot, R_leftfoot, Jv_leftfoot, Jw_leftfoot) = self.chain_left_foot.fkin(qlast[self.joint_indicies_by_chain("pelvis_to_left_foot")])

        #FKIN ON KINEMATIC CHAIN FOR RIGHT FOOT
        (ptip_rightfoot, R_rightfoot, Jv_rightfoot, Jw_rightfoot) = self.chain_right_foot.fkin(qlast[self.joint_indicies_by_chain("pelvis_to_right_foot")])

        #CREATIG JACOBIAN(6 x 48) AND VELOCITY(6 x 1) FOR LEFT FOOT
        J_leftfoot = self.fill_jac(np.vstack((Jv_leftfoot, Jw_leftfoot)), "pelvis_to_left_foot")
        v_leftfoot = np.vstack((vd_leftfoot, wd_leftfoot))

        #CREATIG JACOBIAN(6 x 48) AND VELOCITY(6 x 1) FOR RIGHT FOOT
        J_rightfoot = self.fill_jac(np.vstack((Jv_rightfoot, Jw_rightfoot)), "pelvis_to_right_foot")
        v_rightfoot = np.vstack((vd_rightfoot, wd_rightfoot))

        J_1 = 1/2*self.fill_jac(np.vstack((Jv_left, Jw_left)), "pelvis_to_left_arm") + 1/2*self.fill_jac(np.vstack((Jv_right, Jw_right)), "pelvis_to_right_arm")
        J_1 = np.vstack((J_1, J_leftfoot, J_rightfoot))
        Jwinv_1 = np.linalg.inv(np.transpose(J_1)@J_1 + 0.0001*np.identity(48)) @ np.transpose(J_1)
        e1_v = ep(pd_1, 1/2*(ptip_left+ptip_right))
        e1_r = eR(Rd_1, (R_left+R_right)/2)
        error_1 = np.vstack((e1_v, e1_r))
        v_1 = np.vstack((vd_1, wd_1))
        v_1 = np.vstack((v_1, v_leftfoot, v_rightfoot))
        error_rightfoot = np.vstack((ep(pd_rightfoot, ptip_rightfoot), eR(Rd_rightfoot, R_rightfoot)))
        error_leftfoot = np.vstack((ep(pd_leftfoot, ptip_leftfoot), eR(Rd_leftfoot, R_leftfoot)))
        error_legs = np.vstack((error_leftfoot, error_rightfoot))
        error_1 = np.vstack((error_1, error_legs))

        J_2 = self.fill_jac(np.vstack((Jv_left, Jw_left)), "pelvis_to_left_arm") - self.fill_jac(np.vstack((Jv_right, Jw_right)), "pelvis_to_right_arm")
        Jwinv_2 = np.linalg.inv(np.transpose(J_2)@J_2 + 0.0001*np.identity(48)) @ np.transpose(J_2)
        e2_v = ep(pd_2, ptip_left - ptip_right)
        e2_r = eR(Rd_2, R_left - R_right)
        error_2 = np.vstack((e2_v, e2_r))
        v_2 = np.vstack((vd_2, wd_2))

        qdot_sec = Jwinv_2 @ (v_2 + self.lam*error_2)
        qdot = (Jwinv_1 @ (v_1 + self.lam*error_1)) + (np.identity(48) - Jwinv_1 @ J_1)@qdot_sec
        q = qlast + dt*qdot
        self.q = q
        self.error = np.vstack((error_1, error_2))
        return (q.flatten().tolist(), qdot.flatten().tolist())
    

        # J = np.vstack((J_1, J_2))
        # Jwinv = np.linalg.inv(np.transpose(J)@J + 0.0001*np.identity(48)) @ np.transpose(J)
        # v = np.vstack((vd_1, vd_2))
        # qdot = Jwinv @ (v + self.lam*error)
        # q = qlast + dt*qdot
        # self.error = np.vstack((error_1, error_2))
        # # print(self.error)
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
