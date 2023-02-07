import os
import socket
import threading
from datetime import datetime

from xavier.core.training import Training as ModelTraining
from xavier.lib.cortex.cortex import Cortex


class Subcribe:
    PORT = 5000
    NUMBER_PRED = 3
    MAX_LIST_SIZE = 30

    def __init__(self, client_id, client_secret, path_model, model_type, ip='', path='', username=''):
        self.streams = ('pow',)
        self.ip = ip
        self.save_folder = "{}/{}".format(path, username)
        self.socket_stream = None
        self.c = Cortex(client_id, client_secret, debug_mode=False)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_pow_data=self.on_new_pow_data)
        self.c.bind(inform_error=self.on_inform_error)

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.save_path = f'{self.save_folder}/predict.txt'

        f = open(self.save_path, "w")

        self.model_type = model_type
        self.path_model = path_model

        self.model_training = ModelTraining()
        self.model_training.load_model(self.model_type, self.path_model)

        self.list_predict = []

        if self.ip != '':
            self.socket_stream = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

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
        self.c.close()

    def sub(self, streams):
        self.c.sub_request(streams)

    def unsub(self, streams):
        self.c.unsub_request(streams)

    def on_new_pow_data(self, *args, **kwargs):
        data = kwargs.get('data')
        feature = data['pow']
        result = self.model_training.predict(feature)
        if result is not None:
            self.list_predict.append(result)
        if len(self.list_predict) > (Subcribe.NUMBER_PRED * 3):
            if int(sum(self.list_predict[-Subcribe.NUMBER_PRED:]) / Subcribe.NUMBER_PRED) == self.list_predict[-1]:
                predict = self.list_predict[-1]
                self.list_predict = []
                print('PREDICT: {}'.format(predict))
                path = f'{self.save_folder}/predict.txt'
                with open(path, "a") as file:
                    file.write(f"{predict}, {datetime.now()}\n")
                if self.socket_stream:
                    process_thread = threading.Thread(target=self.send_data, args=(predict,))
                    process_thread.start()
            elif len(self.list_predict) > Subcribe.MAX_LIST_SIZE:
                self.list_predict = []
                if self.socket_stream:
                    process_thread = threading.Thread(target=self.send_data, args=(-1,))
                    process_thread.start()

    def send_data(self, result):
        try:
            self.socket_stream.sendto(str(result).encode('utf8'), (self.ip, Subcribe.PORT))
        except Exception as ex:
            print(f'{ex}')

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
