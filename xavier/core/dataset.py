import numpy as np
import torch
import torch.utils.data

from xavier.constants.type import Type


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, model_type):
        temp_x = []

        if model_type == Type.cnn:
            temp_closter_x = []
            for index, x_row in enumerate(x):
                arr = x_row.reshape((5, 5))
                temp_x.append(np.asarray(arr))

                if index > 3:
                    temp = np.asarray([temp_x[index - 2], temp_x[index - 1], temp_x[index]])
                    temp_closter_x.append(temp)

            self.data = torch.from_numpy(np.asarray(temp_closter_x))

            if y.size != 0:
                y = y[4:]
        elif model_type == Type.rnn or model_type == Type.chrononet:
            for x_row in x:
                arr = x_row.reshape((5, 5))
                temp_x.append(arr)
            self.data = torch.from_numpy(np.asarray(temp_x))

        self.labels = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
