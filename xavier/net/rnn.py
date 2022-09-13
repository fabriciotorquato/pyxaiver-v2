import math

import numpy as np
import torch
from torch import nn

from xavier.constants.type import Type
from xavier.core.transformation import get_standard


class Rnn(nn.Module):
    NAME_TYPE = Type.rnn

    def __init__(self, device=None):
        super(Rnn, self).__init__()
        self.input_layer = 121
        self.matriz_size = int(math.sqrt(self.input_layer))
        self.input_size = 11
        self.hidden_size = 16
        self.num_layers = 2
        self.output_layer = 3
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_layer).to(self.device)

    def forward(self, x):
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        x, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        x = self.fc(x[:, -1, :]).to(self.device)
        return torch.softmax(x, dim=1).to(self.device)

    def convert_standard(self, feature):
        x_row = np.asarray(get_standard([feature], self.standard))[0]
        arr = np.zeros(self.matriz_size * self.matriz_size - len(x_row))
        arr = np.append(x_row, arr, axis=0)
        data_standard = np.asarray(arr).reshape((self.matriz_size, self.matriz_size))
        data_standard = np.array([data_standard])
        return data_standard
