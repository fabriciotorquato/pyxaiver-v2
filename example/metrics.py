from argparse import ArgumentParser

from xavier.core.metrics import Metrics


def get_args():
    parser = ArgumentParser(description='Xavier')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    path = args.path

    predict = f'{path}/predict.txt'
    random = f'{path}/data_random.txt'
    classification = f'{path}/classification.txt'
    image_list = f'{path}/dataList.csv'
    dataset = Metrics(predict, classification, random, image_list, path)
