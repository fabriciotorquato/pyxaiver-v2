import os
from argparse import ArgumentParser
from glob import glob

from xavier.builder.dataset import Dataset


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    path = args.path

    filenames = glob(os.path.join(path, "*", ""))
    dataset = None

    print(filenames)
    for idx, file in enumerate(filenames):
        print(file)
        if file.rpartition('/')[0].rpartition('/')[2] != 'full':
            values = '{}values.csv'.format(file)
            random = '{}data_random.txt'.format(file)
            classification = '{}classification.txt'.format(file)
            dataList = '{}dataList.csv'.format(file)
            dataset = Dataset(values, classification, random, dataList, file)

    if dataset is not None:
        dataset.merge_files(path, filenames)
