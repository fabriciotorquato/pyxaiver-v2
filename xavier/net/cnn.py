import numpy as np
import torch.nn as nn

from xavier.constants.type import Type
from xavier.core.transformation import get_standard


class Cnn(nn.Module):
    NAME_TYPE = Type.cnn

    def __init__(self, device=None):
        super(Cnn, self).__init__()
        self.feature_cnn = []
        self.device = device
        self.output_layer = 3
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=1, padding=1),
            nn.BatchNorm2d(8, False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 4, stride=1, padding=1),
            nn.BatchNorm2d(16, False),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 1 * 1, self.output_layer, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv2(x)
        x = x.view(-1, 16 * 1 * 1)
        x = self.classifier(x)
        return x

    def convert_standard(self, feature):
        x_row = np.asarray(get_standard([feature], self.standard))[0]
        data_standard = np.asarray(x_row).reshape((5, 5))

        if len(self.feature_cnn) > 2:
            self.feature_cnn = self.feature_cnn[1:]
            self.feature_cnn.append(data_standard)
            data_standard = np.array([self.feature_cnn])
            return data_standard
        else:
            self.feature_cnn.append(data_standard)
            return None
