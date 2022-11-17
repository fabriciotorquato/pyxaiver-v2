export $(grep -v '^#' .env | xargs)

sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} 'rm -rf ~/ros/catkin_ws/src'
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/ROS/sender/ ${MACHINE_NAME}@${MACHINE_IP}:~/ros/catkin_ws/src/
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/scripts/run_1_ros.sh ${MACHINE_NAME}@${MACHINE_IP}:~/ros/catkin_ws/
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/scripts/run_2_ros.sh ${MACHINE_NAME}@${MACHINE_IP}:~/ros/catkin_ws/
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/.env ${MACHINE_NAME}@${MACHINE_IP}:~/ros/catkin_ws/


sshpass -p raspberry ssh ${RASP_NAME}@${RASP_IP} 'rm -rf ~/ros/catkin_ws/src'
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/ROS/sender/ ${RASP_NAME}@${RASP_IP}:~/ros/catkin_ws/src/
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/scripts/run_rasp.sh ${RASP_NAME}@${RASP_IP}:~/ros/catkin_ws/
