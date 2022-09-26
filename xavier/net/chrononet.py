import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xavier.constants.type import Type
from xavier.core.transformation import get_standard


class Inception(nn.Module):

    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 32, 1, stride=2)
        self.conv3 = nn.Conv1d(in_channels, 32, 6, stride=2, padding=3)

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        cat = [x1, x2, x3]
        x = torch.cat(cat, dim=1)
        return x


class ChronoNet(nn.Module):
    NAME_TYPE = Type.chrononet

    def __init__(self, device=None):
        super(ChronoNet, self).__init__()
        self.device = device
        self.input_layer = 64
        self.matriz_size = int(math.sqrt(self.input_layer))
        self.output_layer = 3
        self.inception1 = Inception(5)
        self.inception2 = Inception(96)
        self.inception3 = Inception(96)
        self.gru1 = nn.GRU(96, 32, num_layers=1, batch_first=True)
        self.affine1 = nn.Linear(2, 1)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.gru4 = nn.GRU(96, 2, batch_first=True)
        self.classifier = nn.Linear(192, self.output_layer, bias=True)

    def forward(self, x):
        x = x.contiguous().view(-1, 5, 5)
        x = self.inception1(x)
        x = self.inception2(x)
        # x = self.inception3(x)
        x = x.contiguous().view(-1, 2, 96)
        x, _ = self.gru1(x)
        x_res = x
        x, _ = self.gru2(x)
        x_res2 = x
        x_cat1 = torch.cat([x_res, x], dim=2)
        x, _ = self.gru3(x_cat1)
        x = torch.cat([x_res, x_res2, x], dim=2)
        x = x.contiguous().view(-1, 96, 2)
        # x = F.elu(self.affine1(x))
        # x = x.contiguous().view(64, 1, 96)
        # x, _ = self.gru4(x)
        # x = torch.squeeze(x, dim=1)

        x = x.view(-1, 192)
        x = self.classifier(x)
        return x

    def convert_standard(self, feature):
        x_row = np.asarray(get_standard([feature], self.standard))[0]
        data_standard = np.asarray(x_row).reshape((5, 5))
        data_standard = np.array([[data_standard]])
        return data_standard
