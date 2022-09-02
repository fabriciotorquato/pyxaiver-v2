# echo-client.py

import socket

HOST = "192.168.0.15"  # The server's hostname or IP address
PORT = 5000  # The port used by the server
result = 'stop'
#result = 'turn left'
#result = 'turn right'
#result = 'forward'
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.send(str(result).encode('utf8'))


