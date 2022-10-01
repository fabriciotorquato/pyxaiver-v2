sshpass -p raspberry ssh jarvis@192.168.0.15 'rm -rf ~/Documents/pyxavier-v2'
sshpass -p raspberry scp -r ~/Documents/pyxavier-v2/ jarvis@192.168.0.15:~/Documents