export $(grep -v '^#' .env | xargs)

sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} 'rm -rf ~/Documents/pyxavier-v2'
sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} "mkdir -p ~/Documents/pyxavier-v2/example"
sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} "mkdir -p ~/Documents/pyxavier-v2/xavier"

sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/example ${MACHINE_NAME}@${MACHINE_IP}:~/Documents/pyxavier-v2
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/xavier ${MACHINE_NAME}@${MACHINE_IP}:~/Documents/pyxavier-v2