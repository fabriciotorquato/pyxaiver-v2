import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 32, 2, stride=2)
        self.conv3 = nn.Conv1d(in_channels, 32, 8, stride=2, padding=3)

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        cat = [x1, x2, x3]
        x = torch.cat(cat, dim=1)
        return x


class ChronoNet(nn.Module):
    def __init__(self):
        super(ChronoNet, self).__init__()
        self.inception1 = Inception(22)
        self.inception2 = Inception(96)
        self.inception3 = Inception(96)
        self.gru1 = nn.GRU(96, 32, num_layers=1, batch_first=True)
        self.affine1 = nn.Linear(1875, 1)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.gru4 = nn.GRU(96, 2, batch_first=True)

    def forward(self, x):
        x = x.contiguous().view(64, 22, 15000)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = x.contiguous().view(64, 1875, 96)
        x, _ = self.gru1(x)
        x_res = x
        x, _ = self.gru2(x)
        x_res2 = x
        x_cat1 = torch.cat([x_res, x], dim=2)
        x, _ = self.gru3(x_cat1)
        x = torch.cat([x_res, x_res2, x], dim=2)
        x = x.contiguous().view(64, 96, 1875)
        x = F.elu(self.affine1(x))
        x = x.contiguous().view(64, 1, 96)
        x, _ = self.gru4(x)
        x = torch.squeeze(x, dim=1)
        x = F.softmax(x)
        return x
