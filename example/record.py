import os
from argparse import ArgumentParser

from dotenv import load_dotenv

from xavier.core.record import Record

load_dotenv()


# load_dotenv('/home/miguel/my_project/.env')


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--username', type=str, default='user_x')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    path = args.path
    username = args.username
    save_folder = "{}/{}".format(path, username)

    your_app_client_id = os.environ.get('EMOTIV_APP_CLIENT_ID', '')
    your_app_client_secret = os.environ.get('EMOTIV_APP_CLIENT_SECRET', '')

    record_eeg = Record(your_app_client_id, your_app_client_secret, save_folder)

    try:
        record_eeg.start()
    except KeyboardInterrupt:
        record_eeg.stop()
    except Exception as ex:
        record_eeg.stop()
        print(ex)
