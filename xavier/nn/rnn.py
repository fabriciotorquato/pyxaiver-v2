import math

from xavier.builder.model import Model
from xavier.constants.type import Type


def rnn(filename, name_type, model_type=Type.rnn, learning_rate=0.001, times=3, num_epoch=5, batch_size=8,
        input_layer=121, output_layer=3):
    matriz_size = int(math.sqrt(input_layer))
    hidden_layer = matriz_size
    model = Model(filename=filename, name_type=name_type, learning_rate=learning_rate,
                  num_epoch=num_epoch, batch_size=batch_size,
                  input_layer=input_layer, matriz_size=matriz_size, hidden_layer=hidden_layer,
                  output_layer=output_layer, model_type=model_type)
    model.create_model(times=times)
    return model.file_accucary
