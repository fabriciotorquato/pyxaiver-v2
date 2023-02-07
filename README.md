# Pyxavier

## Setup Project

### Main Machine

    pip install -r requirements.txt

### ROS Machine

#### Install ROS

    [ROS](http://wiki.ros.org/kinetic/Installation)
    roscore
    sudo apt-get install ros-indigo-turtlesim

#### make the workplace

    mkdir -p ~/ros/catkin_ws/src    
    cd ~/ros/catkin_ws  
    catkin_make
    Copy ROS/sender/turtlesim_cleaner in ~/ros/catkin_ws/src

# Run Train Model

## Order

Command | Moviment
--- | --- 
0 | up
1 | left
2 | right

## Main Machine

    ./scripts/sync_xavier.sh

    ./scripts/sync_ROS.sh

    python -m example.record --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_6" --username="user_12"

    python -m example.image_player --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_6" --username="user_12" --times_image=1 --wait_time=5 --classification_time=50

    python -m example.create_dataset --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_6"

    ./scripts/send_dataset.sh exp_6_full

    ./scripts/remote_training_model.sh exp_6_full exp_6.csv

    or    

    python -m example.training_model --dir="exp_6_full" --filename=exp_6.csv

    ./scripts/get_model.sh exp_6

# Run Metrics of Model

## Raspberry

    cd ~/ros/catkin_ws
    ./run_rasp.sh

## ROS Machine, first terminal

    cd ~/ros/catkin_ws
    ./run_1_ros.sh

    # typing 'i' for start
    # typing 'p' for stop

## ROS Machine, second terminal

    cd ~/ros/catkin_ws
    ./run_2_ros.sh

## Main Machine

### Test model

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/exp_6/rnn/V13 0.71.pkl" --type_nn="rnn" --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_6" --username="rnn"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/exp_6/cnn/V13 0.53.pkl" --type_nn="cnn" --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_7" --username="cnn"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/exp_6/chrononet/V13 0.86.pkl" --type_nn="chrononet" --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_10" --username="chrononet"

### Run metrics

    python -m example.image_player --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_6" --username="rnn" --times_image=20 --wait_time=1 --classification_time=5

    python -m example.image_player --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_6" --username="cnn" --times_image=20 --wait_time=1 --classification_time=5

    python -m example.image_player --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_9" --username="chrononet" --times_image=20 --wait_time=1 --classification_time=5

### Evaluate

    python -m example.metrics --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_6/rnn"
    
    python -m example.metrics --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_6/cnn"
    
    python -m example.metrics --path="/Users/ftl/Documents/pyxavier-v2/metrics/exp_10/chrononet"
