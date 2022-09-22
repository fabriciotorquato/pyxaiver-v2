import os
import sys
from argparse import ArgumentParser

try:
    from xavier.core.record import Record
except:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from xavier.core.record import Record


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

    your_app_client_id = 'yc0M0hL4rOEwcj8hcB7tuyqNe5Snzfeh4d9R5Eru'
    your_app_client_secret = 'b9vnOoWJp8i1Qh0JTXNMZ9glb9N1Qk0fVE9fVtQtbBFXwdPuP0GfbbXEsgAwlfzUurXVtCVmZVld4E6lmN7j4QgXT0xjaFDoUaLXhQSuhPFa82j21wZymQVs4u4kh0WF'

    record_eeg = Record(your_app_client_id, your_app_client_secret, save_folder)

    try:
        record_eeg.start()
    except KeyboardInterrupt:
        record_eeg.stop()
    except Exception as ex:
        record_eeg.stop()
        print(ex)
