import numpy as np
from torch import nn

from xavier.constants.type import Type
from xavier.core.transformation import get_standard


class Rnn(nn.Module):
    NAME_TYPE = Type.rnn

    def __init__(self, device=None):
        super(Rnn, self).__init__()
        self.output_layer = 3
        self.device = device
        self.lstm = nn.LSTM(5, 16, 2, batch_first=True)
        self.classifier = nn.Linear(16 * 5, self.output_layer, bias=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.contiguous().view(-1, 16, 5)
        x = x.view(-1, 16 * 5)
        x = self.classifier(x)
        return x

    def convert_standard(self, feature):
        x_row = np.asarray(get_standard([feature], self.standard))[0]
        data_standard = np.asarray(x_row).reshape((5, 5))
        data_standard = np.array([data_standard])
        return data_standard
