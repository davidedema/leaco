<p align='center'>
    <h1 align="center">LEArning for COntrol</h1>
    <p align="center">
    Project for Learning and Optimization for Robot Control at the University of Trento A.Y.2024/2025
    </p>
</p>

----------

- [Project Description](#project-description)
- [Project structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation-1)
- [Running the project](#running-the-project)


## Project Description

The aim of this project is to learn an approximate control-invariant set that can be used as terminal
constraint in an MPC formulation. The systems used are double pendulum and UR5

## Project structure
```
learning_for_control
├── double_pendulum
│   ├── build_dataset.py
│   ├── conf_doublep.py
│   ├── datasets
│   ├── main_mpc.py
│   ├── model.py
│   ├── models
│   │   └── model.pt
│   ├── plot_utils.py
│   ├── robot_simulator.py
│   ├── robot_wrapper.py
│   └── train.py
├── README.md
├── requirements.txt
└── ur5
    ├── build_dataset.py
    ├── conf_ur5.py
    ├── datasets
    ├── main_mpc.py
    ├── model.py
    ├── models
    │   └── model.pt
    ├── plot_utils.py
    ├── robot_simulator.py
    ├── robot_wrapper.py
    └── train.py

```
The entrypoint is `main_mpc.py` for both systems
## Installation
### Prerequisites
The prerequisites are the one listed in the main [orc](https://github.com/andreadelprete/orc) repository.

This could be also runned in docker with the following command
```
docker run  -v /tmp/.X11-unix/:/tmp/.X11-unix/ --volume <volume/you/want/to/mount>:/home/student/shared --name ubuntu_bash --env="DISPLAY=$DISPLAY" --privileged -p 127.0.0.1:7000:7000 --shm-size 2g --rm -i -t --user=student --workdir=/home/student andreadelprete/orc24:ass3 bash
```
### Installation
In order to run the project you'll need to clone it and install the requirements. 
- Clone it

    ```BASH
    git clone https://github.com/davidedema/leaco

    ```
- Then go inside the cloned directiory and install all the dependencies
  ```
  cd leaco
  pip install -r requirements.txt
  ```

  If inside in docker remember to commit the dockerimage before exiting the container!
    #### PERMANENT CHANGES IN DOCKER
    ```
    docker ps
    ```
    copy container_id
    ```
    docker commit <container_id> andreadelprete/orc24:ass3
    ```
    now for docker run andreadelprete/orc24:ass3 instead of andreadelprete/orc24:v1

## Running the project
For running the main script enter in the folder `double_pendulum` or `ur5` and
```
python3 main_mpc.py
```
In order to use the Neural Network terminal contraint just set the `USE_TERMINAL_CONSTRAINT` flag to true

New dataset could be generated with the `generate_dataset.py` script and for training them just use the `train.py` script.

