#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import socket

def talker():
    pub = rospy.Publisher('command', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1)   
    HOST = ''              
    PORT = 5000           
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    orig = (HOST, PORT)
    tcp.bind(orig)
    tcp.listen(1)
    con = None
    while not rospy.is_shutdown():
        con, cliente = tcp.accept()
        msg = con.recv(1024)
        msg=str(msg, 'utf8')
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()
    if con is not None:
        con.close()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
