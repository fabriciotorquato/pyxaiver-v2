from argparse import ArgumentParser

from xavier.builder.model import Model
from xavier.net.cnn import Cnn
from xavier.net.eggnet import EEGNet
from xavier.net.rnn import Rnn


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--dir', type=str, default='bci')
    parser.add_argument('--filename', type=str, default='user_a.csv')
    args = parser.parse_args()
    return args


def start_train(filename):
    results = []

    model = Model(filename=filename,
                  learning_rate=0.001,
                  num_epoch=35,
                  batch_size=8,
                  model_cls=Rnn)
    model.create_model(times=1)
    results.append(model.file_accucary)

    model = Model(filename=filename,
                  learning_rate=0.0003,
                  num_epoch=25,
                  batch_size=128,
                  model_cls=Cnn)
    model.create_model(times=1)
    results.append(model.file_accucary)

    model = Model(filename=filename,
                  learning_rate=0.001,
                  num_epoch=50,
                  batch_size=32,
                  model_cls=EEGNet)
    model.create_model(times=1)
    results.append(model.file_accucary)

    for idx, result in enumerate(results):
        if idx == 0:
            print("RNN -> {:.3f}".format(result))
        elif idx == 1:
            print("CNN -> {:.3f}".format(result))
        elif idx == 2:
            print("EEGNET -> {:.3f}".format(result))


if __name__ == "__main__":
    args = get_args()

    name_type = args.dir
    filename = args.filename

    filename = 'dataset/{}/{}'.format(name_type, filename)
    start_train(filename)
