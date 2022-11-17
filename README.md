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

# Run

## Order

Command | Moviment
--- | --- 
0 | up
1 | left
2 | right

## Main Machine

    ./scripts/sync_xavier.sh

    ./scripts/sync_ROS.sh

    python -m example.record --path="/Users/ftl/Documents/pyxavier-v2/dataset/test" --username="exp_4"

    python -m example.image_player --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_4" --username="test_1"

    python -m example.create_dataset --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_4"

    ./scripts/send_dataset.sh exp_4_full

    ./scripts/remote_training_model.sh exp_4_full exp_4.csv

    or    

    python -m example.training_model --dir="exp_4_full" --filename=exp_4.csv

    ./scripts/get_model.sh exp_4

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

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/exp_4/rnn/V13 0.72.pkl" --type_nn="rnn"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/exp_4/cnn/V13 0.57.pkl" --type_nn="cnn"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/exp_4/chrononet/V13 0.86.pkl" --type_nn="chrononet"