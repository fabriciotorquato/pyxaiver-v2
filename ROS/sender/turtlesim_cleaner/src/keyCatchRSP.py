#!/usr/bin/env python3

import rospy
from std_msgs.msg import String


def talker():
    pub = rospy.Publisher('key_catch', String, queue_size=10)
    rospy.init_node('letterTalkerS', anonymous=True)
    rate = rospy.Rate(1)  # 1hz

    while True:
        if rospy.is_shutdown():
            break
        something = input()
        rospy.loginfo(something)
        pub.publish(something)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    except Exception as ex:
        print(ex)
