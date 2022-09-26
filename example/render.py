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
    parser.add_argument('--ip', type=str, default="192.168.0.15")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    model = args.model
    model_type = args.type_nn
    ip = args.ip

    if model_type == "eegnet":
        model_type = Type.eegnet
    elif model_type == "rnn":
        model_type = Type.rnn
    elif model_type == "cnn":
        model_type = Type.cnn
    elif model_type == "chrononet":
        model_type = Type.chrononet

    your_app_client_id = 'yc0M0hL4rOEwcj8hcB7tuyqNe5Snzfeh4d9R5Eru'
    your_app_client_secret = 'b9vnOoWJp8i1Qh0JTXNMZ9glb9N1Qk0fVE9fVtQtbBFXwdPuP0GfbbXEsgAwlfzUurXVtCVmZVld4E6lmN7j4QgXT0xjaFDoUaLXhQSuhPFa82j21wZymQVs4u4kh0WF'
    subcribe = Subcribe(your_app_client_id, your_app_client_secret, model, model_type, ip)

    try:
        subcribe.start()
    except KeyboardInterrupt:
        subcribe.stop()
    except Exception as ex:
        subcribe.stop()
        print(ex)
