# figure_skating_robot

Commands: <br />
cd ~/robotws <br />
rm -rf build install log <br />
source /opt/ros/humble/setup.bash <br />

cd ~/robotws <br />
colcon build --symlink-install <br />
source ~/robotws/install/setup.bash <br />

cd ~/robotws/install/packages/hw3code/rviz <br />
ros2 run rviz2 rviz2 -d viewurdf.rviz <br />

cd ~/robotws/install/packages/figure_skating_robot <br />
ros2 run robot_state_publisher robot_state_publisher humanSubject06_48dof.urdf <br />

ros2 run joint_state_publisher_gui joint_state_publisher_gui <br />

ros2 run figure_skating_robot PreliminaryTesting.py <br />
