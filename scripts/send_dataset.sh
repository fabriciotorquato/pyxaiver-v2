export $(grep -v '^#' .env | xargs)

sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} 'rm -rf ~/Documents/pyxavier-v2/dataset'
sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} 'rm -rf ~/Documents/pyxavier-v2/models'
sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} "mkdir -p ~/Documents/pyxavier-v2/dataset/$1"

sshpass -p raspberry scp -r  ~/Documents/pyxavier-v2/dataset/$1 ${MACHINE_NAME}@${MACHINE_IP}:~/Documents/pyxavier-v2/dataset