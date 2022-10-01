import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn

import xavier.constants.config as config
from xavier.core.data_loader import DataLoader
from xavier.core.dataset import Dataset
from xavier.core.training import Training
from xavier.core.transformation import get_standard, init_standard
from xavier.net.cnn import Cnn

torch.manual_seed(1234)


class Model(object):

    def __init__(self, filename, learning_rate, num_epoch, batch_size, model_cls=Cnn):
        self.filename = filename
        self.model_cls = model_cls
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.file_accucary = np.zeros(1)
        self.version = config.VERSION
        self.model_training = None

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.set_device(0)
        print("Algorithim use: ", self.device)

        self.filename_model = "models/{}/{}".format(self.model_cls.NAME_TYPE.value,
                                                    filename.split('.')[-2].split('/')[-1])
        self.filename_model = os.path.abspath(self.filename_model)
        Path(self.filename_model).mkdir(parents=True, exist_ok=True)

    def __train_model(self, train_loader, valid_loader, test_loader, train_size, valid_size, filename):
        for epoch in range(self.num_epoch):
            print("Epoch {}/{}".format(epoch + 1, self.num_epoch + 1))
            epoch_train_loss = self.model_training.train(train_loader, train_size)
            epoch_validate_loss = self.model_training.validation(valid_loader, valid_size)
        print('Data train:')
        self.model_training.validation(train_loader, train_size, validation=False)
        print('Data Test:')
        return self.model_training.test(test_loader, filename)

    def __build_model_training(self):
        model = self.model_cls(self.device).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        model = model.to(self.device)
        model_training = Training(model, criterion, optimizer, self.device)
        return model_training

    def get_dataset(self):
        dataset = np.loadtxt(self.filename, skiprows=1, delimiter=',', dtype=np.float64)
        dataset = self._remove_outliers(dataset)
        dataset_x = np.asarray([l[1:] for l in dataset])
        dataset_y = np.asarray([l[0] for l in dataset])
        return dataset_x, dataset_y

    def get_normalization(self, x_train, x_test):
        x_train, standard = init_standard(x_train)
        x_test = get_standard(x_test, standard)
        return x_train, x_test, standard

    def get_loader(self, x_train, y_train, x_test, y_test):
        train_data = Dataset(x_train, y_train, self.model_cls.NAME_TYPE, self.model_training.model.input_layer)
        test_data = Dataset(x_test, y_test, self.model_cls.NAME_TYPE, self.model_training.model.input_layer)

        data_loader = DataLoader()
        train_loader, valid_loader, train_size, valid_size = data_loader.get_train(train_data, self.batch_size)
        test_loader = data_loader.get_test(test_data, self.batch_size)
        return train_loader, valid_loader, test_loader, train_size, valid_size

    def save_model(self, acc, standard):
        filename = '{}/{} {:.2f}.pkl'.format(self.filename_model, self.version, acc)
        torch.save({'model': self.model_training.model.state_dict(), 'standard': standard}, filename)

    def _remove_outliers(self, data):
        filter_dataset = np.array(data[::])
        for col in range(1, len(filter_dataset.T)):
            a = np.array(filter_dataset.T[col])
            upper_quartile = np.percentile(a, 95)
            lower_quartile = np.percentile(a, 0)
            IQR = (upper_quartile - lower_quartile) * 1.5
            quartile_set = (lower_quartile - IQR, upper_quartile + IQR)
            result_list = []
            for idx_row, y in enumerate(filter_dataset):
                if quartile_set[0] <= y[col] <= quartile_set[1]:
                    result_list.append(filter_dataset[idx_row])
            filter_dataset = np.array(result_list)
        return filter_dataset

    def create_model(self, times):
        print("Training dataset: {}".format(self.filename))
        self.file_accucary = np.zeros(1)
        for _ in range(times):
            self.model_training = self.__build_model_training()
            dataset_x, dataset_y = self.get_dataset()

            x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=21)
            x_train, x_test, standard = self.get_normalization(x_train, x_test)
            train_loader, valid_loader, test_loader, train_size, valid_size = self.get_loader(x_train, y_train,
                                                                                              x_test, y_test)

            filename = "{}/{}".format(self.filename_model, self.version)
            accuracy = self.__train_model(train_loader, valid_loader, test_loader, train_size, valid_size, filename)

            if accuracy > self.file_accucary:
                self.save_model(accuracy, standard)
                self.file_accucary = accuracy
