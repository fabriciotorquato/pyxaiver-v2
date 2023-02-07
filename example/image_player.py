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
    parser.add_argument('--times_image', type=int, default=1)
    parser.add_argument('--wait_time', type=int, default=5)
    parser.add_argument('--classification_time', type=int, default=50)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    path = args.path
    username = args.username
    save_folder = "{}/{}".format(path, username)
    times_image = args.times_image
    wait_time = args.wait_time
    classification_time = args.classification_time

    try:
        app_image_bci_player.main(path, username, times_image, wait_time, classification_time)
    except Exception as ex:
        print(ex)
