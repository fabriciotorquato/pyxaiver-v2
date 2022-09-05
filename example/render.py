import os
import sys
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
    model_type = args.type_nn
    path = args.path
    username = args.username
    train = args.train
    url = args.url

    if model_type == "mlp":
        model_type = Type.mlp
    elif model_type == "rnn":
        model_type = Type.rnn
    elif model_type == "cnn":
        model_type = Type.cnn

    try:
        your_app_client_id = ''
        your_app_client_secret = ''
        s = Subcribe(your_app_client_id, your_app_client_secret, model, model_type, train, url)
        streams = ['pow']
        s.start(streams)
    except Exception as ex:
        print(ex)
