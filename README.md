# figure_skating_robot

Commands:
cd ~/robotws
rm -rf build install log
source /opt/ros/humble/setup.bash

cd ~/robotws 
colcon build --symlink-install 
source ~/robotws/install/setup.bash 

cd ~/robotws/install/packages/hw3code/rviz 
ros2 run rviz2 rviz2 -d viewurdf.rviz

cd ~/robotws/install/packages/figure_skating_robot
ros2 run robot_state_publisher robot_state_publisher humanSubject06_48dof.urdf 

ros2 run joint_state_publisher_gui joint_state_publisher_gui

ros2 run figure_skating_robot PreliminaryTesting.py