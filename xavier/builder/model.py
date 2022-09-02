import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn

import xavier.constants.config as config
from xavier.constants.type import Type
from xavier.core.dataLoader import DataLoader
from xavier.core.dataset import Dataset
from xavier.core.training import Training
from xavier.core.transformation import get_standard, init_standard
from xavier.net.cnn import Cnn
from xavier.net.mlp import Mlp
from xavier.net.rnn_lstm_2 import Rnn

torch.manual_seed(1234)


class Model(object):

    def __init__(self, filename, name_type, learning_rate, num_epoch, batch_size, input_layer=0,
                 hidden_layer=0, output_layer=3, matriz_size=0, model_type=Type.mlp):

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.matriz_size = matriz_size
        self.filename = filename
        self.matriz_size = matriz_size
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.file_accucary = np.zeros(1)
        self.version = config.VERSION
        self.model = None

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.set_device(0)
        print("Algorithim use: ", self.device)

        self.filename_model = "models/{}/{}".format(name_type, filename.split('.')[-2].split('/')[-1])
        self.filename_model = os.path.abspath(self.filename_model)
        Path(self.filename_model).mkdir(parents=True, exist_ok=True)

    def __train_model(self, train_loader, valid_loader, test_loader, train_size, valid_size, filename):
        self.model.first_time = time.time()
        for epoch in range(self.num_epoch):
            print("Epoch {}/{}".format(epoch + 1, self.num_epoch + 1))
            self.model.train(train_loader, train_size)
            self.model.validation(valid_loader, valid_size)
        self.model.validation(train_loader, train_size, validation=False)
        return self.model.test(test_loader, filename)

    def __build_model(self):
        if self.model_type == Type.mlp:
            model = Mlp(self.device).to(self.device)
        elif self.model_type == Type.rnn:
            model = Rnn(self.device).to(self.device)
            print(model)
        elif self.model_type == Type.cnn:
            model = Cnn(self.device).to(self.device)
        else:
            return None

        # choose optimizer and loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        model = model.to(self.device)
        model_training = Training(model, criterion, optimizer, self.device)
        return model_training

    def get_dataset(self):
        dataset = np.loadtxt(self.filename, skiprows=1, delimiter=',', dtype=np.float64)
        dataset_x = np.asarray([l[1:] for l in dataset])
        dataset_y = np.asarray([l[0] for l in dataset])
        return dataset_x, dataset_y

    def get_normalization(self, x_train, x_test):
        if self.model_type == Type.mlp:
            x_train, standard = init_standard(x_train)
            x_test = get_standard(x_test, standard)
            return x_train, x_test, standard
        elif self.model_type == Type.rnn:
            x_train, standard = init_standard(x_train)
            x_test = get_standard(x_test, standard)
            return x_train, x_test, standard
        elif self.model_type == Type.cnn:
            x_train, standard = init_standard(x_train)
            x_test = get_standard(x_test, standard)
            return x_train, x_test, standard

    def get_loader(self, x_train, y_train, x_test, y_test):
        if self.model_type == Type.mlp:
            train_data = Dataset(x_train, y_train, Type.mlp)
            test_data = Dataset(x_test, y_test, Type.mlp)
        elif self.model_type == Type.rnn:
            train_data = Dataset(x_train, y_train, Type.rnn, self.matriz_size)
            test_data = Dataset(x_test, y_test, Type.rnn, self.matriz_size)
        elif self.model_type == Type.cnn:
            train_data = Dataset(x_train, y_train, Type.cnn, self.matriz_size)
            test_data = Dataset(x_test, y_test, Type.cnn, self.matriz_size)
        else:
            return

        data_loader = DataLoader()
        train_loader, valid_loader, train_size, valid_size = data_loader.get_train(train_data, self.batch_size)
        test_loader = data_loader.get_test(test_data, self.batch_size)
        return train_loader, valid_loader, test_loader, train_size, valid_size

    def save_model(self, acc, standard):
        path = "{}/{}".format(self.filename_model, self.model_type.value)
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = '{}/{} {:.2f}.pkl'.format(path, self.version, acc)
        torch.save({'model': self.model.model.state_dict(), 'standard': standard}, filename)

    def create_model(self, times):
        print("Training dataset: {}".format(self.filename))
        self.file_accucary = np.zeros(1)
        for _ in range(times):
            dataset_x, dataset_y = self.get_dataset()
            x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=21)
            x_train, x_test, standard = self.get_normalization(x_train, x_test)
            train_loader, valid_loader, test_loader, train_size, valid_size = self.get_loader(x_train,
                                                                                              y_train,
                                                                                              x_test,
                                                                                              y_test)

            self.input_layer = x_train.shape[1]
            self.model = self.__build_model()
            if self.model is None:
                return

            filename = "{}/{}".format(self.filename_model, self.version)

            accuracy = self.__train_model(train_loader, valid_loader, test_loader, train_size, valid_size, filename)

            if accuracy > self.file_accucary:
                self.save_model(accuracy, standard)
                self.file_accucary = accuracy
