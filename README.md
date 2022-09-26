# Pyxavier

## Library

### Setup project

    pip install -r requirements.txt

### Run

    python -m example.record --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_2" --username="user_4"

    python -m example.image_player --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_2" --username="user_4"

    python -m example.create_dataset --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_2" 

    python -m example.training_model --dir="exp_2_full" --filename=exp_2.csv

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/eegnet/exp_2/V13 0.46.pkl" --type_nn="eegnet" --ip="192.168.0.15"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/rnn/exp_2/V13 0.52.pkl" --type_nn="rnn" --ip="192.168.0.15"