export $(grep -v '^#' .env | xargs)

export ROS_MASTER_URI=http://${RASP_IP}:11311
export ROS_IP=${MACHINE_IP}

cd ~/ros/catkin_ws
source devel/setup.bash
rosrun turtlesim_cleaner talker2.py
