# Description of the Implementation

## Learning Algorithm

The model used for training the agent was an Actor-Critic network which a local and target network for both the Actor and Critic.
So overall, we have four neural networks that update periodically. The architecture of the networks is very similar and can be described as follows:

- Input Layer: State size (which happens to be 33 in our case)
- First Hidden Layer: 256
- Second Hidden Layer: 128
- Output Layer: Action-Size (which in our case is 4 and is also continuous)

To train our agent, we use a replay buffer/replay memory to store the experiences that the agen has with the environment after every timestep.
The replay memory has a size of 10^6 which means that we can store 10^6 experience tuples (state, action, reward, next_state, done) before it is full.

Once the replay memory has greater than 1024 experience tuples, at every 20th timestep, we sample a batch of 1024 tuples to train with. The agent then
learns from these experiences and updates its weights for both the actor and the critic.
