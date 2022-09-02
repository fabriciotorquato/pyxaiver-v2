import json
import ssl
from xavier.lib.cortex.cortex import Cortex
from xavier.core.training import Training as ModelTraining
import websocket

class Subcribe:
    def __init__(self, app_client_id, app_client_secret, path_model,type, isTrain=False, url="wss://localhost:6868", **kwargs):
        self.c = Cortex(app_client_id, app_client_secret, debug_mode=True, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_dev_data=self.on_new_dev_data)
        self.c.bind(new_pow_data=self.on_new_pow_data)
        self.c.bind(inform_error=self.on_inform_error)

        self.type = type
        self.path_model = path_model
        self.isTrain = isTrain

        if url:
            self.ws = websocket.create_connection(url, sslopt={"cert_reqs": ssl.CERT_NONE})
        else:
            self.ws = None
            
        if not self.isTrain:
            self.init_model()

    def init_model(self):
        try:
            self.model_training = ModelTraining()
            self.model_training.load_model(self.type, self.path_model)
        except Exception as ex:
            print('{}'.format(ex))

    def start(self, streams, headsetId=''):
        self.streams = streams

        if headsetId != '':
            self.c.set_wanted_headset(headsetId)

        self.c.open()

    def sub(self, streams):
        self.c.sub_request(streams)

    def unsub(self, streams):
        self.c.unsub_request(streams)

    def on_new_data_labels(self, *args, **kwargs):
        data = kwargs.get('data')
        stream_name = data['streamName']
        stream_labels = data['labels']
        print('{} labels are : {}'.format(stream_name, stream_labels))


    def on_new_dev_data(self, *args, **kwargs):
        data = kwargs.get('data')
        print('dev data: {}'.format(data))

    def on_new_pow_data(self, *args, **kwargs):
        data = kwargs.get('data')
        print('pow data: {}'.format(data))


        if not self.isTrain:
            feature=data['pow']
            #feature = get_feature(delta, theta, alpha, beta)
            result = self.model_training.predict(feature)

            print('value: {}'.format(result))

            if self.ws:
                try:
                    QUERY_RECORD_ID = 100
                    query_predict_request = {
                        "jsonrpc": "2.0",
                        "method": "queryPredict",
                        "params": {
                            "predict": result,
                        },
                        "id": QUERY_RECORD_ID
                    }
                    self.ws.send(json.dumps(query_predict_request))
                    result = self.ws.recv()
                except Exception as ex:
                    print('{}'.format(ex))

    def on_create_session_done(self, *args, **kwargs):
        print('on_create_session_done')
        self.sub(self.streams)

    def on_inform_error(self, *args, **kwargs):
        error_data = kwargs.get('error_data')
        print(error_data)

