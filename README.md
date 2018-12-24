# DeepRL-Nanodegree-Project2 (ContinuousControl)

In this project, we will train a Deep Deterministic Policy Gradient (DDPG) agent to try and solve Unity's Reacher environment.

### Environment Description



In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic and in order to solve the environment, our agent must achieve an average score of +30 over 100 consecutive episodes.

### Download Instructions

Here are the instructions to folllow if you'd like to try out this agent on your machine. First you'll need at least Python 3.6 installed on your system. You will also need these libraries to help run the code. Most of these can be installed using the 'pip install' command on your terminal once Python has been installed.

1. numpy - NumPy is the fundamental package for scientific computing with Python
1. collections - High-performance container datatypes
1. torch - PyTorch is an optimized tensor library for deep learning using GPUs and CPUs
1. unityagents - Unity Machine Learning Agents allows researchers and developers to transform games and simulations created using the Unity Editor into environments where intelligent agents can be trained using reinforcement learning, evolutionary strategies, or other machine learning methods through a simple to use Python API
1. matplotlib.pyplot - Provides a MATLAB-like plotting framework

You can download the environment from one of the links below. You need only select the environment that matches your operating system:
#### Version 1: One (1) Agent
  - Linux: click here
  - Mac OSX: click here
  - Windows (32-bit): click here
  - Windows (64-bit): click here
#### Version 2: Twenty (20) Agents
  - Linux: click here
  - Mac OSX: click here
  - Windows (32-bit): click here
  - Windows (64-bit): click here
  
 
### File Descriptions

The repo contains 6 main files:
1. ContinuousControl_Train.py - This file, written in Python 3.6 with the help of the PyTorch framework contains the ddpg function that we use to train the agent with. It runs until the agent has solved the environment which can vary between 150 - 200 episodes depending on the hyperparameter selection.

1. model.py - This file consists the Actor-Critic model.

1. ddpg_agent.py - This file consists the ddpg_agent class and its methods/functions to interact with and observe the environment.

1. ContinuousControl_Test.py - This file can be used to test the trained agent. It runs for a total of 10 episodes and plots the performance in each of them.

1. checkpoint_actor.pth - This file consists the agent's trained Actor weights. You may use this file if you'd like to use the pretrained agent to solve the environment. This file also gets recreated every time you run the ContinuousControl_Train.py file. So you can create your own checkpoint.pth file with your choice of hyperparameters!

1. checkpoint_critic.pth - This file consists of the agen'ts trained Critic weights. This also gets recreated every time you run ContinuousControl_Train.py.

### How to run the code?

- Clone/download the 6 files listed above and add them in the same folder as the Reacher environment on your machine. You can run the code using a terminal like Anaconda Prompt or anything that can run python commands like 'pip.
- Once you navigate to the folder where the project files are located using the 'cd' command, run either 'ContinuousContorl_Train.py' file if you'd like to train your own agent or 'ContinuousControl_Test.py' if you'd like to see a pretrained agent in action!

Please refer to the Report.md file if you'd like an in-depth look of the architecture.
