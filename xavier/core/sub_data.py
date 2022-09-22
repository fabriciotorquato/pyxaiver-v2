import socket
import threading

from xavier.core.training import Training as ModelTraining
from xavier.lib.cortex.cortex import Cortex


class Subcribe:
    PORT = 5000

    def __init__(self, client_id, client_secret, path_model, model_type, ip="192.168.0.15"):
        self.streams = ('pow',)
        self.ip = ip
        self.socket_stream = None
        self.c = Cortex(client_id, client_secret, debug_mode=True)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_pow_data=self.on_new_pow_data)
        self.c.bind(inform_error=self.on_inform_error)

        self.model_type = model_type
        self.path_model = path_model

        self.model_training = ModelTraining()
        self.model_training.load_model(self.model_type, self.path_model)

        if self.ip != "":
            self.socket_stream = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            # self.socket_stream.bind((self.ip, Subcribe.PORT))

    def start(self):
        print("Start Render")
        try:
            self.c.open()
        except Exception as ex:
            print(ex)

    def stop(self):
        print("Stop Record")
        self.unsub(self.streams)
        if self.socket_stream:
            try:
                self.socket_stream.close()
            except Exception as ex:
                print(ex)

    def sub(self, streams):
        self.c.sub_request(streams)

    def unsub(self, streams):
        self.c.unsub_request(streams)

    def on_new_pow_data(self, *args, **kwargs):
        data = kwargs.get('data')

        feature = data['pow']
        # feature = get_feature(delta, theta, alpha, beta)
        result = self.model_training.predict(feature)
        if self.socket_stream:
            process_thread = threading.Thread(target=self.send_data, args=(result,))
            process_thread.start()

    def send_data(self, result):
        try:
            self.socket_stream.sendto(str(result).encode('utf8'), (self.ip, Subcribe.PORT))
            print('send value: {}'.format(result))
        except Exception as ex:
            print('{}'.format(ex))

    def on_create_session_done(self, *args, **kwargs):
        self.sub(self.streams)

    def on_inform_error(self, *args, **kwargs):
        error_data = kwargs.get('error_data')
        print(error_data)
        try:
            if self.c.session_id != '':
                self.unsub(self.streams)
        except Exception as ex:
            print(ex)
            raise KeyboardInterrupt
