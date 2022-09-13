import numpy as np
import torch
from torch import nn

from xavier.core.transformation import get_standard


class Mlp(nn.Module):
    def __init__(self, device=None):
        super(Mlp, self).__init__()
        self.device = device
        self.output_layer = 3
        self.dense = nn.Sequential(
            nn.Linear(112, 32).to(self.device),
            nn.Tanh().to(self.device),
            nn.Linear(32, 32).to(self.device),
            nn.Tanh().to(self.device),
            nn.Linear(32, self.output_layer).to(self.device),
        )

    def forward(self, x):
        x = self.dense(x).to(self.device)
        return torch.softmax(x, dim=1).to(self.device)

    def convert_standard(self, feature):
        data_standard = np.asarray(get_standard([feature], self.standard))[0]
        data_standard = np.array([data_standard])
        return data_standard
