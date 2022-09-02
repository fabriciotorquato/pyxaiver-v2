#!/usr/bin/env python
import rospy
import requests
from std_msgs.msg import String

def talker():
    apiUrl = "http://192.168.0.20:8080/xavier"
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 1hz
    while not rospy.is_shutdown():
        response = requests.get(apiUrl)
        data = response.json()
        hello_str = data["state"]
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
