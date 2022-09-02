#!/usr/bin/env python
import rospy
import requests
from std_msgs.msg import String
from SunFounder_Ultrasonic_Avoidance import Ultrasonic_Avoidance

def talker():
    pub = rospy.Publisher('rpSender', String, queue_size=10)
    rospy.init_node('rspPi', anonymous=True)
    rate = rospy.Rate(1) # 1hz
    while not rospy.is_shutdown():
        distance = UA.get_distance()
        status = UA.less_than(threshold)
        pub.publish(distance)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
