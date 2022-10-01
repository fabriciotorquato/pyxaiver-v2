sshpass -p raspberry ssh jarvis@192.168.0.15 'rm -rf ~/ros/catkin_ws/src'
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/ROS/sender/ jarvis@192.168.0.15:~/ros/catkin_ws/src/

sshpass -p raspberry ssh pi@192.168.0.38 'rm -rf ~/ros/catkin_ws/src'
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/ROS/sender/ pi@192.168.0.38:~/ros/catkin_ws/src/
