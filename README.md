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

## Main Machine

    python -m example.record --path="/Users/ftl/Documents/pyxavier-v2/dataset/test" --username="exp_1"

    python -m example.image_player --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_4" --username="user_2"

    python -m example.create_dataset --path="/Users/ftl/Documents/pyxavier-v2/dataset/exp_4" 

    python -m example.training_model --dir="exp_4_full" --filename=exp_4.csv

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/eegnet/exp_4/V13 0.49.pkl" --type_nn="eegnet" --ip="192.168.0.15"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/rnn/exp_4/V13 0.65.pkl" --type_nn="rnn" --ip="192.168.0.15"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/cnn/exp_4/V13 0.56.pkl" --type_nn="cnn" --ip="192.168.0.15"

    python -m example.render --model="/Users/ftl/Documents/pyxavier-v2/models/chrononet/exp_4/V13 0.84.pkl" --type_nn="chrononet" --ip="192.168.0.15"

## Raspberry

    cd ~/ros/catkin_ws
    source devel/setup.bash
    rosrun turtlesim_cleaner picarROS.py

## ROS Machine, first terminal

    export ROS_MASTER_URI=http://192.168.0.38:11311
    export ROS_IP=192.168.0.15

    cd ~/ros/catkin_ws
    source devel/setup.bash
    rosrun turtlesim_cleaner keyCatchRSP.py

## ROS Machine, second terminal

    export ROS_MASTER_URI=http://192.168.0.38:11311
    export ROS_IP=192.168.0.15

    cd ~/ros/catkin_ws
    source devel/setup.bash
    rosrun turtlesim_cleaner talker2.py
