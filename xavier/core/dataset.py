import math

import numpy as np
import torch
import torch.utils.data

from xavier.constants.type import Type


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, model_type=Type.mlp, input_layer=0):
        matriz_size = int(math.sqrt(input_layer))

        temp_x = []

        if model_type == Type.cnn:
            temp_closter_x = []
            for index, x_row in enumerate(x):
                arr = np.zeros(matriz_size * matriz_size - len(x_row))
                arr = np.append(x_row, arr, axis=0)
                arr = arr.reshape((matriz_size, matriz_size))
                temp_x.append(np.asarray(arr))

                if index > 3:
                    temp = []
                    temp.append(temp_x[index - 2])
                    temp.append(temp_x[index - 1])
                    temp.append(temp_x[index])
                    temp = np.asarray(temp).reshape((3, matriz_size, matriz_size))
                    temp_closter_x.append(np.asarray(temp))

            self.data = torch.from_numpy(np.asarray(temp_closter_x))

            if y.size != 0:
                y = y[4:]

        elif model_type == Type.rnn:
            for x_row in x:
                arr = np.zeros(matriz_size * matriz_size - len(x_row))
                arr = np.append(x_row, arr, axis=0)
                arr = arr.reshape((matriz_size, matriz_size))
                temp_x.append(np.asarray(arr))
            self.data = torch.from_numpy(np.asarray(temp_x))

        self.labels = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
