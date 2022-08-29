import os
from glob import glob
from xavier.nn.cnn import cnn
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--show', '-s', action='store_true')
    parser.add_argument('--full', '-f', action='store_false')
    parser.add_argument('--dir', '-d', type=str, default='bci')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    name_type = args.dir
    show = args.show
    full = args.full

    path_to_directory = 'dataset/'+name_type+'/'
    if full:
        filenames = [path_to_directory+"full/"]
    else:
        filenames = glob(os.path.join(path_to_directory, "*", ""))

    result = cnn(filenames=filenames, name_type=name_type, show=show, times=1, output_layer=3)

    print("CNN:")
    print("->", result, "\n")
