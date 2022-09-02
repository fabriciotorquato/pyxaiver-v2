from xavier.builder.model import Model
from xavier.constants.type import Type


def mlp(filename, name_type, model_type=Type.mlp, learning_rate=0.001, times=3, num_epoch=5, batch_size=16,
        input_layer=112, hidden_layer=32, output_layer=3):
    model = Model(filename=filename, name_type=name_type,
                  learning_rate=learning_rate, num_epoch=num_epoch, batch_size=batch_size,
                  input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer, model_type=model_type)
    model.create_model(times=times)
    return model.file_accucary
