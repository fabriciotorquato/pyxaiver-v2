import csv

import numpy as np


def get_time_classification(path):
    file_csv = open(path)
    data_csv = csv.reader(file_csv)
    data_csv = np.array([np.array(row) for row in data_csv])
    timestamps = []
    for row in data_csv:
        temp = row[0].split(" ")
        timestamps.append(temp[0] + " " + temp[3] + " " + temp[4])
    return timestamps
