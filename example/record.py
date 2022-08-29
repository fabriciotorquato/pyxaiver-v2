import datetime
import sys
import os
from argparse import ArgumentParser
try:
    from xavier.core.record import Record
except:
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))
    from xavier.core.record import Record


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--type_nn', type=str, default='')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--username', type=str, default='')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--ip', type=str, default='127.0.0.1:8080')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()

    model = args.model
    type_nn = args.type_nn
    path = args.path
    username = args.username
    train = args.train
    ip = args.ip

    try:
        # Please fill your application clientId and clientSecret before running script
        your_app_client_id = ''
        your_app_client_secret = ''

        r = Record(your_app_client_id, your_app_client_secret)
        r.record_title = 'values_%s.csv' % str(datetime.datetime.now()).replace(':', '-')
        r.record_export_folder = "{}/{}".format(path,username)
        r.record_export_data_types = ['BP']
        r.record_export_format = 'CSV'
        r.record_export_version = 'V2'

        record_duration_s = 10  # duration for recording in this example. It is not input param of create_record
        r.start(record_duration_s)
    except Exception as ex:
      print(ex)