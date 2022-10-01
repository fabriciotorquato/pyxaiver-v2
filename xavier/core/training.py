import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import torch
from sklearn import metrics
from torch.autograd import Variable

from xavier.constants.type import Type
from xavier.net.chrononet import ChronoNet
from xavier.net.cnn import Cnn
from xavier.net.eggnet import EEGNet
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
        self.model_type = None

    def load_model(self, model_type, path_model):
        self.model_type = model_type

        if self.model_type == Type.mlp:
            self.model = Mlp()
        elif self.model_type == Type.rnn:
            self.model = Rnn()
        elif self.model_type == Type.cnn:
            self.model = Cnn()
        elif self.model_type == Type.eegnet:
            self.model = EEGNet()
        elif self.model_type == Type.chrononet:
            self.model = ChronoNet()

        checkpoint = torch.load(path_model, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.standard = checkpoint['standard']
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
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
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

    def predict(self, feature):
        try:
            data_standard = self.model.convert_standard(feature)
            if data_standard is None:
                return None
            self.model.zero_grad()
            data_standard = torch.LongTensor(data_standard)
            data_standard = Variable(data_standard).float()
            output = self.model(data_standard)
            top_prob, top_label = torch.topk(output, self.model.output_layer)
            return top_label[0][0].item()
        except Exception as ex:
            print('{}'.format(ex))
