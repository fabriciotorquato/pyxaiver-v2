sshpass -p raspberry ssh jarvis@192.168.0.15 'rm -rf ~/Documents/pyxavier-v2/dataset'
sshpass -p raspberry ssh jarvis@192.168.0.15 'rm -rf ~/Documents/pyxavier-v2/models'
sshpass -p raspberry scp -r  ~/Documents/pyxavier-v2/dataset/exp_4_full jarvis@192.168.0.15:~/Documents/pyxavier-v2/dataset