export $(grep -v '^#' .env | xargs)

sshpass -p raspberry scp -r ${MACHINE_NAME}@${MACHINE_IP}:~/Documents/pyxavier-v2/models/$1 ~/Documents/pyxavier-v2/models