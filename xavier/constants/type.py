from enum import Enum


class Type(Enum):
    mlp = "mlp"
    rnn = "rnn"
    cnn = "cnn"
    eegnet = "eegnet"
    chrononet = "chrononet"
