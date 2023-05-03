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

    python -m example.record --path="/Users/fabricioleite/Documents/pyxavier-v2/dataset/exp_8" --username="user_1"

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/dataset/exp_8" --username="user_1" --times_image=1 --wait_time=5 --classification_time=50

    python -m example.create_dataset --path="/Users/fabricioleite/Documents/pyxavier-v2/dataset/exp_7"

    ./scripts/send_dataset.sh exp_7_full

    ./scripts/remote_training_model.sh exp_7_full exp_7.csv

    or    

    python -m example.training_model --dir="exp_7_full" --filename=exp_7.csv

    ./scripts/get_model.sh exp_7

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

    python -m example.render --model="/Users/fabricioleite/Documents/pyxavier-v2/models/exp_7/rnn/V13 0.71.pkl" --type_nn="rnn" --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7_3" --username="rnn"

    python -m example.render --model="/Users/fabricioleite/Documents/pyxavier-v2/models/exp_7/cnn/V13 0.54.pkl" --type_nn="cnn" --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7_1" --username="cnn"

    python -m example.render --model="/Users/fabricioleite/Documents/pyxavier-v2/models/exp_7/chrononet/V13 0.82.pkl" --type_nn="chrononet" --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7_4" --username="chrononet"

### Run metrics

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7_1" --username="rnn" --times_image=20 --wait_time=1 --classification_time=5

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7_1" --username="cnn" --times_image=20 --wait_time=1 --classification_time=5

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7_3" --username="chrononet" --times_image=20 --wait_time=1 --classification_time=5

### Evaluate

    python -m example.metrics --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7/rnn"
    
    python -m example.metrics --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7/cnn"
    
    python -m example.metrics --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/exp_7/chrononet"
