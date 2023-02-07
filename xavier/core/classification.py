import csv
import time
from datetime import datetime

import numpy as np


class Classification(object):

    def __init__(self, file_csv, classification, save_folder):
        self.saveFolder = save_folder
        self.seconds = 0
        self.raw_data = self.csv_modification(file_csv)
        self.time_raw_data = self.get_time_raw_data(file_csv)
        self.timestamps = self.get_time_classification(classification)
        self.feature = self.find_feature()

    def get_time_raw_data(self, path):
        with open(path) as file_csv:
            data_csv = csv.reader(file_csv)
            next(data_csv, None)
            next(data_csv, None)
            data_csv = np.asarray([int(float(row[0])) for row in data_csv])
        return data_csv.ravel()

    def get_time_classification(self, path):
        with open(path) as file_data:
            list_files = list(file_data)
        timestamps = np.array(list_files).reshape(len(list_files) // 2, 2)
        return timestamps

    def csv_modification(self, path):
        with open(path) as file_csv:
            data_csv = csv.reader(file_csv)
            next(data_csv, None)
            next(data_csv, None)
            data_csv = np.asarray([np.array(row[1:]).astype(np.float64) for row in data_csv])
        return data_csv

    def get_csv(self, path):
        with open(path) as file_csv:
            data_csv = csv.reader(file_csv)
            data_training = np.array([each_line for each_line in data_csv])
        return data_training

    def find_feature(self):
        feature = []
        for idx, timestamps in enumerate(self.timestamps):
            feature.append([])
            begin_time = timestamps[0].strip()
            begin_time = int(time.mktime(datetime.strptime(begin_time, "%Y-%m-%d %H:%M:%S").timetuple()))

            end_time = timestamps[1].strip()
            end_time = int(time.mktime(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timetuple()))

            for index_raw_data, value in enumerate(self.time_raw_data):
                if begin_time <= value <= end_time:
                    feature[idx].append(self.raw_data[index_raw_data])
        return feature
