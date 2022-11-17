import os
from argparse import ArgumentParser

from dotenv import load_dotenv

from xavier.constants.type import Type
from xavier.core.sub_data import Subcribe

load_dotenv()


# load_dotenv('/home/miguel/my_project/.env')


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--type_nn', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    model = args.model
    model_type = args.type_nn

    if model_type == "rnn":
        model_type = Type.rnn
    elif model_type == "cnn":
        model_type = Type.cnn
    elif model_type == "chrononet":
        model_type = Type.chrononet

    your_app_client_id = os.environ.get('EMOTIV_APP_CLIENT_ID', '')
    your_app_client_secret = os.environ.get('EMOTIV_APP_CLIENT_SECRET', '')
    ip = os.environ.get('MACHINE_IP', '')
    subcribe = Subcribe(your_app_client_id, your_app_client_secret, model, model_type, ip)

    try:
        subcribe.start()
    except KeyboardInterrupt:
        subcribe.stop()
    except Exception as ex:
        subcribe.stop()
        print(ex)
