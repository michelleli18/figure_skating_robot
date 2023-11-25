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
from GeneratorNode      import GeneratorNode
from TransformHelpers   import *
from TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from KinematicChain     import KinematicChain


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        #LEFT SIDE JOINTS
        #Initial joint positions
        self.qo_left = np.radians(np.array([]).reshape(-1,1))
       
        #Set up left kinematic chain object
        self.chain_left = KinematicChain(node, 'world', 'tip', self.jointnames())

        #RIGHT SIDE JOINTS
        #Initial joint positions
        self.q0_right = np.radians(np.array([0, 90, -90, 0, 0, 0]).reshape((-1,1)))

        # Set up right kinematic chain object.
        self.chain_right = KinematicChain(node, 'world', 'tip', self.jointnames())

        self.p0 = np.array([0.0, 0.55, 1.0]).reshape((3,1))
        self.R0 = Reye()
        self.pleft  = np.array([0.3, 0.5, 0.15]).reshape((3,1))
        self.pright = np.array([-0.3, 0.5, 0.15]).reshape((3,1))
        self.phigh = np.array([0.0, 0.5, 0.9]).reshape((3,1))
        
        self.q  = self.q0
        self.error = np.zeros((6,1))
        self.lam = 20


    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        if t < 3:
            #going from the initial position to the first target
            (g0, g0dot) = goto(t, 3.0, 0.0, 1.0) 
            #p0 = 0, pf = 1. This 0 and 1 is with respect to the predermined path. 
            #0 being the beginning of this path. 1 being the right of the path. 
            #And -1 being the left end of the path
            pd = self.p0 + (self.pright - self.p0) * g0
            vd =           (self.pright - self.p0) * g0dot


            Rd = Reye()
            wd = np.zeros((3,1))
        else:
            #Now we choose a path. That path will be a parabolic shape between the two ends.
            #The position and velocities the tip goes through will be a function of the path g.
            #The way in which the path g is traversed, which is independent of the actual pos and vel
            #of the tip, will be a function of time.
            g = cos(2*pi/5 * (t-3.0)) #2pi/5 because we want entire movement to take 5s
            gdot = -2*pi/5 * sin(2*pi/5 * (t-3.0))

            t1 = (t-3) % 5.0
            if t1 < 2.5:
                #We want the starting orientation to be like that from g = 0 and end at g = 1
                #If we make it from -1 to 1 (like in 4), then the rotation path will be symmetric.
                #We dont want this since the right and left orientation arent a mirror of each other.
                (gR, gRdot) = goto(t1,     2.5, 0.0, 1.0)
            else:
                (gR, gRdot) = goto(t1-2.5, 2.5, 1.0, 0.0)

            pd = np.array([-0.3 * g,    0.50,0.9 - 0.75*g**2]).reshape((3,1))
            vd = np.array([-0.3 * gdot, 0, -2*0.75*g*gdot]).reshape((3,1))

            Rd = Roty(-pi/2 * gR) @ Rotz(pi/2 * gR)
            #Rd = Rote((ey()), -pi/2 * gR)
            wd = (ey()-ez())*(-pi/2 * gRdot)
        
        qlast = self.q
        error = self.error

        (plast, R, Jv, Jw) = self.chain.fkin(qlast)

        J = np.vstack((Jv, Jw))
        v = np.vstack((vd, wd))

        qdot = np.linalg.inv(J) @ (v + self.lam*error)
        q = qlast + dt*qdot

        self.error = np.vstack((ep(pd, plast), eR(Rd, R)))
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
