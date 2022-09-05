import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from xavier.core.classification import Classification
from xavier.core.transformation import get_feature, get_frequency


class Dataset(object):

    def __init__(self, file_csv, classification, random_list, image_list, save_folder):

        self.file_csv = file_csv
        self.classification = classification
        self.random_list = random_list
        self.image_list = image_list
        self.save_folder = save_folder

        self.classification = Classification(
            self.file_csv, self.classification, self.save_folder)

        self._create_dataset()

    def _get_labels(self, random_list, image_list, seconds):
        with open(random_list) as file_csv:

            data_csv_random_list = csv.reader(file_csv)
            data_csv_random_list = np.array(
                [np.array(row) for row in data_csv_random_list])

        with open(image_list) as file_csv:

            data_csv_image_list = csv.reader(file_csv)
            data_csv_image_list = np.array(
                [np.array(row) for row in data_csv_image_list])
            data_csv_image_list = data_csv_image_list.T
            labels = []

            for row in data_csv_random_list:
                value = row[0].split(" ", 1)[1]
                index = data_csv_image_list[1].tolist().index(value)
                for _ in range(seconds):
                    labels.append(data_csv_image_list[0][index])

            return labels

    def _create_dataset(self):
        array_data = self.classification.get_many_seconds()

        print('{}:'.format(self.save_folder.rpartition('/')[0].rpartition('/')[2]))

        labels = self._get_labels(
            self.random_list, self.image_list, self.classification.seconds)

        with open(self.save_folder + 'dataset.csv', 'w') as dataset_file:
            for index, data in enumerate(tqdm(array_data)):
                if len(data) > 0:
                    data = map(list, zip(*data))
                    (delta, theta, alpha, beta) = get_frequency(data)
                    wave_data = get_feature(delta, theta, alpha, beta)
                    wr = csv.writer(dataset_file)
                    wave_data_with_label = np.insert(wave_data, 0, labels[index])
                    wr.writerow(wave_data_with_label)

    def merge_files(self, save_folder, filenames):

        print("Create Full Dataset:")
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        total = 0
        for sample in filenames:
            if sample.rpartition('/')[0].rpartition('/')[2] != 'full':
                total += sum(1 for row in open('{}dataset.csv'.format(sample)))

        with open('{}dataset.csv'.format(save_folder), 'w') as file_out:
            for sample in filenames:
                if sample.rpartition('/')[0].rpartition('/')[2] != 'full':
                    wr = csv.writer(file_out)

                    file_csv = open(sample + 'dataset.csv')
                    file_csv = csv.reader(file_csv)
                    file_csv = np.array([np.array(row) for row in file_csv])

                    for line in file_csv:
                        wr.writerow(line)
