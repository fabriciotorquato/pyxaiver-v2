from picar import front_wheels
from picar import back_wheels
import xavier_command
import picar

class XavierCar(object):
    def __init__(self):
        # 0 = random direction, 1 = force left, 2 = force right, 3 = orderdly
        self.force_turning = 0
        picar.setup()

        self.fw = front_wheels.Front_Wheels(db='config')
        self.bw = back_wheels.Back_Wheels(db='config')
        self.fw.turning_max = 45

        self.forward_speed = 40
        self.backward_speed = 40

        self.back_distance = 10
        self.turn_distance = 20

        self.timeout = 10
        self.last_angle = 90
        self.last_dir = 0
        self.command = xavier_command.STOP

    def start_avoidance(self):
        self._update_status()

    def send_command(self, command):
        self.command = command.data
        print(command.data)
        self._update_status()

    def _update_status(self):
        if self.command == xavier_command.FORWARD:
            self.fw.turn_straight()
            self.bw.backward()
            self.bw.speed = self.forward_speed
        elif self.command == xavier_command.TURN_LEFT:
            self.fw.turn(30)
            self.bw.backward()
            self.bw.speed = self.forward_speed
        elif self.command == xavier_command.TURN_RIGHT:
            self.fw.turn(150)
            self.bw.backward()
            self.bw.speed = self.forward_speed
        elif self.command == xavier_command.STOP:
            self.fw.turn_straight()
            self.bw.stop()


    def stop(self):
        self.bw.stop()
        self.fw.turn_straight()
