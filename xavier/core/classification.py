import csv
import time
from datetime import datetime

import numpy as np

import xavier.constants.eeg as eeg


class Classification(object):

    def __init__(self, file_csv, classification, save_folder):
        self.saveFolder = save_folder
        self.seconds = 0
        self.raw_data = self.csv_modification(file_csv)
        self.time_raw_data = self.get_time_raw_data(file_csv)
        self.timestamps = self.get_time_classification(classification)
        self.array_index = self.find_index()

    def get_time_raw_data(self, path):
        with open(path) as file_csv:
            data_csv = csv.reader(file_csv)
            next(data_csv, None)
            next(data_csv, None)
            data_csv = np.asarray([int(float(row[0])) for row in data_csv])
        return data_csv.ravel()

    def get_time_classification(self, path):
        with open(path) as file_csv:
            data_csv = csv.reader(file_csv)
            timestamps = np.array([np.array(row)[0] for row in data_csv])
        return timestamps

    def csv_modification(self, path):
        with open(path) as file_csv:
            data_csv = csv.reader(file_csv)
            next(data_csv, None)
            next(data_csv, None)
            data_csv = np.asarray([np.array(row[2:]).astype(np.float64) for row in data_csv])
        return data_csv

    def get_csv(self, path):
        with open(path) as file_csv:
            data_csv = csv.reader(file_csv)
            data_training = np.array([each_line for each_line in data_csv])
        return data_training

    def find_index(self):
        array_index = []
        index_timestamps = 0
        for index_raw_data, value in enumerate(self.time_raw_data):
            time_2 = self.timestamps[index_timestamps].split(" ", 1)[1]
            time_2 = int(time.mktime(datetime.strptime(time_2, "%Y-%m-%d %H:%M:%S").timetuple()))
            if time_2 < value:
                array_index.append(max(0, index_raw_data - 1))
                index_timestamps = index_timestamps + 1
                if index_timestamps == len(self.timestamps):
                    break
        return array_index

    def get_many_seconds(self):
        feature = []
        for index in self.array_index:
            step_index = 0
            while eeg.SAMPLING_RATE + eeg.WINDOW_SIZE * step_index < eeg.SECONDS_RECORD * eeg.SAMPLING_RATE:
                feature.append(
                    self.raw_data[
                    index + step_index * eeg.WINDOW_SIZE:index + eeg.SAMPLING_RATE + step_index * eeg.WINDOW_SIZE])
                step_index += 1
            self.seconds = step_index
        return feature
