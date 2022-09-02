import datetime
import sys
import os
from argparse import ArgumentParser

from xavier.constants.type import Type

try:
    from xavier.core.sub_data import Subcribe
except:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from xavier.core.sub_data import Subcribe


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--type_nn', type=str, default='')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--username', type=str, default='')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--url', type=str, default='wss://localhost:6868')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()

    model = args.model
    type_nn = args.type_nn
    path = args.path
    username = args.username
    train = args.train
    url = args.url

    if type_nn == "mlp":
        type_nn = Type.mlp
    elif type_nn == "rnn":
        type_nn = Type.rnn
    elif type_nn == "cnn":
        type_nn = Type.cnn

    try:
        your_app_client_id = ''
        your_app_client_secret = ''
        s = Subcribe(your_app_client_id, your_app_client_secret, model, type_nn, train, url)
        streams = ['pow']
        s.start(streams)
    except Exception as ex:
      print(ex)