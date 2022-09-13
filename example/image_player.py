import os
import sys
from argparse import ArgumentParser

try:
    from player import app_image_bci_player
except:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from player import app_image_bci_player


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
    try:
        app_image_bci_player.main(path, username)
    except Exception as ex:
        print(ex)
