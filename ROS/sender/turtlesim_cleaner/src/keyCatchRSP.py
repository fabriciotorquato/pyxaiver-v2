#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import xavier_command

def talker():
    initi = True
    pub = rospy.Publisher('command', String, queue_size=10)
    rospy.init_node('letterTalkerS', anonymous=True)
    rate = rospy.Rate(1) # 1hz
    while not rospy.is_shutdown():
        something = input()
        rospy.loginfo(something)
        pub.publish(something)
        rate.sleep()
    

if __name__ == '__main__':
    try:
        talker()
        car_publisher = rospy.Publisher('command', String, queue_size=10)
        decoy = xavier_command.STOP
        car_publisher.publish(decoy)
    except rospy.ROSInterruptException:
        car_publisher = rospy.Publisher('command', String, queue_size=10)
        decoy =  xavier_command.STOP
        car_publisher.publish(decoy)
        pass

