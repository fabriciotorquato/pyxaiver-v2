#!/usr/bin/env python3
import socket

import rospy
from std_msgs.msg import String

try:

    HOST = ''
    PORT = 5000
    udp = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp.bind((HOST, PORT))


    def myhook():
        udp.close()


    rospy.on_shutdown(myhook)
    pub = rospy.Publisher('command', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(16)

    while True:
        if rospy.is_shutdown():
            break
        data, _ = udp.recvfrom(1024)
        if not data:
            break
        msg = str(data, 'utf8')
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()
except rospy.ROSInterruptException:
    pass
except Exception as ex:
    print(ex)
