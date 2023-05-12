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

# Generate commands
    
    ./scripts/generate_commands.sh {user_name}


## Main Machine

    ./scripts/sync_xavier.sh

    ./scripts/sync_ROS.sh

    python -m example.record --path="/Users/fabricioleite/Documents/pyxavier-v2/dataset/tcc_1" --username="user_13"

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/dataset/tcc_1" --username="user_13" --times_image=1 --wait_time=5 --classification_time=50

    python -m example.create_dataset --path="/Users/fabricioleite/Documents/pyxavier-v2/dataset/tcc_1"

    ./scripts/send_dataset.sh tcc_1_full

    ./scripts/remote_training_model.sh tcc_1_full tcc_1.csv

    or    

    python -m example.training_model --dir="tcc_1_full" --filename=tcc_1.csv

    ./scripts/get_model.sh tcc_1

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
    python -m example.render --model="/Users/fabricioleite/Documents/pyxavier-v2/models/tcc_1/chrononet/V13 0.78.pkl" --type_nn="chrononet" --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1" --username="chrononet"

    python -m example.render --model="/Users/fabricioleite/Documents/pyxavier-v2/models/tcc_1/rnn/V13 0.70.pkl" --type_nn="rnn" --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1" --username="rnn"

    python -m example.render --model="/Users/fabricioleite/Documents/pyxavier-v2/models/tcc_1/cnn/V13 0.62.pkl" --type_nn="cnn" --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1" --username="cnn"


### Run metrics

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1" --username="chrononet" --times_image=20 --wait_time=1 --classification_time=5

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1" --username="rnn" --times_image=20 --wait_time=1 --classification_time=5

    python -m example.image_player --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1" --username="cnn" --times_image=20 --wait_time=1 --classification_time=5

### Evaluate

    python -m example.metrics --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1/rnn"
    
    python -m example.metrics --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1/cnn"
    
    python -m example.metrics --path="/Users/fabricioleite/Documents/pyxavier-v2/metrics/tcc_1/chrononet"
