from argparse import ArgumentParser

from xavier.nn.cnn import cnn
from xavier.nn.mlp import mlp
from xavier.nn.rnn import rnn


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--dir', type=str, default='bci')
    parser.add_argument('--filename', type=str, default='user_a.csv')
    args = parser.parse_args()
    return args


def start_train(filename, name_type):
    results = []

    results.append(mlp(filename=filename, name_type=name_type, times=1, output_layer=3))
    results.append(rnn(filename=filename, name_type=name_type, times=1, output_layer=3))
    results.append(cnn(filename=filename, name_type=name_type, times=1, output_layer=3))

    for idx, result in enumerate(results):
        if idx == 0:
            print("MLP -> {:.3f}".format(result))
        elif idx == 1:
            print("RNN -> {:.3f}".format(result))
        elif idx == 2:
            print("CNN -> {:.3f}".format(result))


if __name__ == "__main__":
    args = get_args()

    name_type = args.dir
    filename = args.filename

    filename = 'dataset/{}/{}'.format(name_type, filename)
    start_train(filename, name_type)
