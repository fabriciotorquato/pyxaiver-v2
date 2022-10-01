import math

import numpy as np
import torch
import torch.nn as nn

from xavier.constants.type import Type
from xavier.core.transformation import get_standard


class Cnn(nn.Module):
    NAME_TYPE = Type.cnn

    def __init__(self, device=None):
        super(Cnn, self).__init__()
        self.device = device
        self.input_layer = 121
        self.matriz_size = int(math.sqrt(self.input_layer))
        self.output_layer = 3
        self.feature_cnn = []
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 8, 4, 1, 1).to(self.device),
            nn.BatchNorm2d(8, False).to(self.device)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 4, 1, 1).to(self.device),
            nn.BatchNorm2d(16, False).to(self.device),
            nn.MaxPool2d(2).to(self.device)
        )
        self.dense = nn.Sequential(
            nn.Linear(8 * 8 * 4, 32).to(self.device),
            nn.Tanh().to(self.device),
            nn.Linear(32, 32).to(self.device),
            nn.Tanh().to(self.device),
            nn.Linear(32, self.output_layer).to(self.device),
        )

    def forward(self, x):
        x = self.conv0(x).to(self.device)
        x = self.conv2(x).to(self.device)
        x = x.view(x.size(0), -1)
        x = self.dense(x).to(self.device)
        return torch.softmax(x, dim=1).to(self.device)

    def convert_standard(self, feature):
        x_row = np.asarray(get_standard([feature], self.standard))[0]
        arr = np.zeros(self.matriz_size * self.matriz_size - len(x_row))
        arr = np.append(x_row, arr, axis=0)
        data_standard = np.asarray(arr).reshape((self.matriz_size, self.matriz_size))

        if len(self.feature_cnn) > 2:
            self.feature_cnn = self.feature_cnn[1:]
            self.feature_cnn.append(data_standard)
            data_standard = np.array([self.feature_cnn])
            return data_standard
        else:
            self.feature_cnn.append(data_standard)
            return None
