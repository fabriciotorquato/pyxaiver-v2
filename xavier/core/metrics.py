import csv
from datetime import datetime
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

from xavier.constants import eeg
from xavier.core.optimise_itr import calcule_itr


class Metrics(object):

    def __init__(self, predict, classification, random_list, image_list, save_file):

        with open(predict) as file_read:
            self.predict = [line.strip() for line in file_read.readlines()]

        with open(classification) as file_read:
            self.classification = [line.strip() for line in file_read.readlines()]

        self.image_list = []

        with open(image_list) as image_read:
            self.legends = [legend.strip() for legend in image_read.readlines()]

        with open(random_list) as file_read:
            for image in file_read.readlines():
                for legend in self.legends:
                    legend, file_name = legend.strip().split(',')
                    if file_name == image.strip():
                        self.image_list.append(legend)
                        break

        self.metrics_file = f'{save_file}/metrics.csv'
        self.save_file = save_file

        self.header_file = ['NCC', 'TA (%)', 'DET', 'TMD (s)', 'TMA (s)', 'ITR (bits/min)']

        self._create_metrics()

    def _create_metrics(self):
        ncc, tdc, det, tma, tmd = self._detection_precision()
        ta = self._find_ta()
        itr = self._itr(ta, tmd)
        self._save_csv_results(ncc, ta * 100, det, tmd, tma, itr)

    def _find_ta(self):
        path_model = self.save_file.replace('/metrics/', '/models/')
        model_filename = [f for f in listdir(path_model) if isfile(join(path_model, f)) and '.pkl' in f][0]
        return float(model_filename.replace(f'{eeg.VERSION} ', '').replace('.pkl', ''))

    def _itr(self, ta, tmd):
        targets = len(self.legends)
        return round(calcule_itr(targets, ta, tmd), 2)

    def _detection_precision(self):
        det = 0
        ncc = 0
        nt = 0

        tmd = []
        tma = []
        fp = {'0': 0, '1': 0, '2': 0}

        for idx, value in enumerate(zip(self.classification[::2], self.classification[1::2])):
            begin, end = value

            real = self.image_list[idx]
            begin_time = None
            old_time_predict = None

            begin = datetime.strptime(begin, "%Y-%m-%d  %H:%M:%S.%f").timestamp()
            end = datetime.strptime(end, "%Y-%m-%d  %H:%M:%S.%f").timestamp()

            for predict in self.predict:
                value, time_predict = predict.split(', ')
                time_predict = datetime.strptime(time_predict, "%Y-%m-%d %H:%M:%S.%f").timestamp()

                if time_predict > end:
                    tma.append(end - begin)
                    break

                if begin_time is None:
                    if time_predict > begin:
                        begin_time = begin
                        old_time_predict = time_predict
                        nt += 1

                if begin_time is not None:
                    det += 1
                    tmd.append(time_predict - old_time_predict)
                    old_time_predict = time_predict
                    if real == value:
                        ncc += 1
                        tma.append(time_predict - begin_time)
                        break
                    else:
                        fp[value] += 1

        # matplotlib histogram
        plt.hist(tma, color='blue', edgecolor='black', bins=int(180 / 5))

        # Add labels
        plt.title('Tempo Medio de Acerto')
        plt.xlabel('Segundos')
        plt.ylabel('Acertos')
        plt.savefig(f'{self.save_file}/histogram_detection_time.png')

        return (
            ncc,
            round((ncc / nt) * 100, 1),
            det,
            round(sum(tma) / len(tma), 3),
            round(sum(tmd) / len(tmd), 3),
        )

    def _save_csv_results(self, ncc, tdc, det, tmd, tma, itr):
        with open(self.metrics_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(self.header_file)
            writer.writerow(
                (ncc, tdc, det, tmd, tma, itr)
            )
