export $(grep -v '^#' .env | xargs)

sshpass -p raspberry ssh ${MACHINE_NAME}@${MACHINE_IP} "cd ~/Documents/pyxavier-v2 && python3 -m example.training_model --dir=$1 --filename=$2"