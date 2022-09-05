import math

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import torch
from sklearn import metrics
from torch.autograd import Variable

from xavier.constants.type import Type
from xavier.core.transformation import get_standard
from xavier.net.cnn import Cnn
from xavier.net.mlp import Mlp
from xavier.net.rnn import Rnn

# Turn interactive plotting off
plt.ioff()


class Training:

    def __init__(self, model=None, criterion=None, optimizer=None, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_tra, self.loss_val, self.feature_cnn = [], [], []
        self.device = device
        self.standard = None

    def load_model(self, model_type, path_model):
        self.model_type = model_type

        if self.model_type == Type.mlp:
            self.model = Mlp()
            checkpoint = torch.load(path_model, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.standard = checkpoint['standard']
            self.model.eval()

        elif self.model_type == Type.rnn:
            self.model = Rnn()
            checkpoint = torch.load(path_model, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.standard = checkpoint['standard']
            self.model.eval()

        elif self.model_type == Type.cnn:
            self.model = Cnn()
            checkpoint = torch.load(path_model, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.standard = checkpoint['standard']
            self.model.eval()

    def train(self, train_loader, size):
        self.model.train()
        test_loss = 0

        for data, target in train_loader:
            data = Variable(data).to(self.device).float()
            target = Variable(target).to(self.device).long()

            self.optimizer.zero_grad()
            output = self.model(data).to(self.device)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            test_loss += loss.item()

        test_loss /= size
        self.loss_tra.append(test_loss)
        return test_loss

    def validation(self, valid_loader, size, validation=True):
        self.model.eval()
        test_loss = 0
        counter = 0
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                counter += 1
                data = Variable(data).to(self.device).float()
                target = Variable(target).to(self.device).long()

                output = self.model(data).to(self.device)
                test_loss += self.criterion(output, target).item()

                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum()

        test_loss /= size

        if validation:
            self.loss_val.append(test_loss)

        print('loss: {:.4f} - acc: {:.2f}'.format(test_loss, correct / size))
        return test_loss

    def test(self, test_loader, filename):
        self.model.eval()
        test_loss = 0
        correct = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for data, target in test_loader:

                data = Variable(data).to(self.device).float()
                target = Variable(target).to(self.device).long()

                output = self.model(data).to(self.device)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum()

                if self.device != "cpu":
                    predicted = predicted.cpu().numpy()
                    target = target.cpu().numpy()

                y_pred.extend(predicted)
                y_true.extend(target)

        test_loss /= len(test_loader.dataset)

        print('loss: {:.4f} - acc: {:.2f}'.format(test_loss, correct / len(test_loader.dataset)))

        test_acc = metrics.accuracy_score(y_true, y_pred)

        fig = plt.figure()
        epochs = np.arange(len(self.loss_tra))
        plt.plot(epochs, self.loss_tra, epochs, self.loss_val)
        plt.savefig('{} {:.2f}_curve.png'.format(filename, test_acc))
        plt.close(fig)

        skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
        plt.savefig('{} {:.2f}_matriz.png'.format(filename, test_acc))

        return np.float32(test_acc)

    def predict_mlp(self, feature):
        return np.asarray(get_standard([feature], self.standard))[0]

    def predict_cnn(self, feature):
        input_layer = 121
        matriz_size = int(math.sqrt(input_layer))
        x_row = np.asarray(
            get_standard([feature], self.standard))[0]
        arr = np.zeros(matriz_size * matriz_size - len(x_row))
        arr = np.append(x_row, arr, axis=0)
        data_standard = np.asarray(arr).reshape((matriz_size, matriz_size))
        return data_standard

    def predict_rnn(self, feature):
        input_layer = 121
        matriz_size = int(math.sqrt(input_layer))
        x_row = np.asarray(
            get_standard([feature], self.standard))[0]
        arr = np.zeros(matriz_size * matriz_size - len(x_row))
        arr = np.append(x_row, arr, axis=0)
        data_standard = np.asarray(arr).reshape((matriz_size, matriz_size))
        return data_standard

    def print_percentage(self, top_label, top_prob):
        for index in range(self.model.output_layer):
            print('option-{}: {}'.format(chr(97 + top_label[0][index].item()), float(top_prob[0][index].item()) * 100))

    def predict(self, feature):
        try:
            if self.model_type == Type.mlp:
                data_standard = np.array([self.predict_mlp(feature)])
            elif self.model_type == Type.rnn:
                data_standard = np.array([self.predict_rnn(feature)])
            elif self.model_type == Type.cnn:
                if len(self.feature_cnn) > 2:
                    self.feature_cnn = self.feature_cnn[1:]
                    self.feature_cnn.append(self.predict_cnn(feature))
                    data_standard = np.array([self.feature_cnn])
                else:
                    self.feature_cnn.append(self.predict_cnn(feature))
                    return int(self.model.output_layer)
            else:
                return

            self.model.zero_grad()
            data_standard = torch.LongTensor(data_standard)
            data_standard = Variable(data_standard).float()
            output = self.model(data_standard)
            top_prob, top_label = torch.topk(output, self.model.output_layer)
            self.print_percentage(top_label, top_prob)

            return top_label[0][0].item()
        except Exception as ex:
            print('{}'.format(ex))
