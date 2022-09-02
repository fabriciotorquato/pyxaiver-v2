#!/usr/bin/env python
import rospy
import requests
from std_msgs.msg import String
import random
from geometry_msgs.msg import Twist
import socket

def talker():
    initi = True
    pub = rospy.Publisher('letter', String, queue_size=10)
    rospy.init_node('letterTalker', anonymous=True)
    rate = rospy.Rate(1) # 1hz
    HOST = '127.0.0.1'              # Endereco IP do Servidor
    PORT = 5000            # Porta que o Servidor esta
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    orig = (HOST, PORT)
    tcp.bind(orig)
    tcp.listen(1)
    con, cliente = tcp.accept()
    print 'Concetado por', cliente
    while not rospy.is_shutdown():
        msg = con.recv(1024)
        if not msg: break
        print cliente, msg
    

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
