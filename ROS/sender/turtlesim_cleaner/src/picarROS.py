#!/usr/bin/env python
from threading import Thread

import rospy
from std_msgs.msg import String
from xavier_car import XavierCar

class TurtleBot:
	
	def __init__(self):
		self.xavier_car = XavierCar()
		self.xavier_car.start_avoidance()
	
	def listener(self):

		rospy.init_node('picar_controller', anonymous=True)
		rospy.Subscriber('command', String, self.xavier_car.send_command)
		rospy.spin()

	def stop(self):
		self.xavier_car.stop()

if __name__ == '__main__':
	turtleBot = TurtleBot()
	try:
		turtleBot.listener()
	except:
		turtleBot.stop()
