import math

import numpy as np
import torch.nn as nn

from xavier.constants.type import Type
from xavier.core.transformation import get_standard


class EEGNet(nn.Module):
    NAME_TYPE = Type.eegnet

    def __init__(self, device=None):
        super(EEGNet, self).__init__()
        self.device = device
        self.T = 128
        self.input_layer = 64
        self.matriz_size = int(math.sqrt(self.input_layer))
        self.output_layer = 3

        self.F1 = 8
        self.F2 = 16
        self.D = 2

        # Conv2d(in,out,kernel,stride,padding,bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 5), padding=(0, 32), bias=False).to(self.device),
            nn.BatchNorm2d(self.F1).to(self.device)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (5, 1), groups=self.F1, bias=False).to(self.device),
            nn.BatchNorm2d(self.D * self.F1).to(self.device),
            nn.ELU().to(self.device),
            nn.AvgPool2d((1, 4)).to(self.device),
            nn.Dropout(0.5).to(self.device)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, 16), padding=(0, 8), groups=self.D * self.F1,
                      bias=False).to(self.device),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False).to(self.device),
            nn.BatchNorm2d(self.F2).to(self.device),
            nn.ELU().to(self.device),
            nn.AvgPool2d((1, 8)).to(self.device),
            nn.Dropout(0.5).to(self.device)
        )

        self.classifier = nn.Linear(32, self.output_layer, bias=True).to(self.device)

    def forward(self, x):
        x = self.conv1(x).to(self.device)
        x = self.conv2(x).to(self.device)
        x = self.Conv3(x).to(self.device)

        x = x.view(-1, 32)
        x = self.classifier(x).to(self.device)
        return x

    def convert_standard(self, feature):
        x_row = np.asarray(get_standard([feature], self.standard))[0]
        data_standard = np.asarray(x_row).reshape((5, 5))
        data_standard = np.array([[data_standard]])
        return data_standard
