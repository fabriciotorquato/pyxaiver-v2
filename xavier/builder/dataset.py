import csv
from pathlib import Path

import numpy as np

from xavier.core.classification import Classification


class Dataset(object):

    def __init__(self, file_csv, classification, random_list, image_list, save_folder):

        self.file_csv = file_csv
        self.classification = classification
        self.random_list = random_list
        self.image_list = image_list
        self.save_folder = save_folder

        self.classification = Classification(self.file_csv, self.classification, self.save_folder)
        self._create_dataset()

    def _get_labels(self):
        with open(self.random_list) as file_random:
            random_list = [row.strip() for row in file_random]

        dict_image = {}
        with open(self.image_list) as file_csv:
            for row in file_csv:
                value, key = row.strip().split(",")
                dict_image[key] = value

        labels = [dict_image[random_value] for random_value in random_list]
        return labels

    def _create_dataset(self):
        feature = self.classification.feature
        labels = self._get_labels()
        with open(self.save_folder + 'dataset.csv', 'w') as dataset_file:
            wr = csv.writer(dataset_file)
            for idx, label in enumerate(labels):
                for data in feature[idx]:
                    wr.writerow(np.insert(data, 0, label))

    def merge_files(self, save_folder, filenames):
        Path(save_folder + '_full/').mkdir(parents=True, exist_ok=True)
        total = 0
        for sample in filenames:
            if sample.rpartition('/')[0].rpartition('/')[2] != 'full':
                total += sum(1 for row in open('{}dataset.csv'.format(sample)))
        with open('{}{}.csv'.format(save_folder + '_full/', save_folder.split('/')[-1]), 'w') as file_out:
            for sample in filenames:
                if sample.rpartition('/')[0].rpartition('/')[2] != 'full':
                    wr = csv.writer(file_out)

                    file_csv = open(sample + 'dataset.csv')
                    file_csv = csv.reader(file_csv)
                    file_csv = np.array([np.array(row) for row in file_csv])

                    for line in file_csv:
                        wr.writerow(line)

        print("Create Full Dataset")
