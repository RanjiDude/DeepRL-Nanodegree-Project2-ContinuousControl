import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from ddpg_agent import Agent

env = UnityEnvironment(file_name='Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# Number of agents in the environment
num_agents = len(env_info.agents)

# Number of actions in the environment
action_size = brain.vector_action_space_size

# Examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

# Create the ddpg agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=3)

# Load the trained agent's weights from checkpoint_actor.pth and checkpoint_critic.pth
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

# Testing the agent for a total of 10 episodes
def test(n_episodes=10):

    scores = []

    for i_episode in range(1, n_episodes+1):
        score = np.zeros(num_agents)                       # Set the score to 0 before the episode begins
        env_info = env.reset(train_mode=False)[brain_name] # Reset the environment
        states = env_info.vector_observations              # Get the current state for each agent
        agent.reset()
        while True:
            actions = agent.act(states)                     # Select an action
            env_info = env.step(actions)[brain_name]        # Take selected actions on the environment
            next_states = env_info.vector_observations      # Get next state from the environment
            rewards = env_info.rewards                      # Get reward for taking selected actions
            dones = env_info.local_done                     # Check to see if the episode has terminated or completed
            states = next_states                            # Set the state as the new_state of the env
            score += rewards                                # Update the scores based on rewards
            if np.any(dones):                               # Break the loop after the episode is done
                scores.append(np.max(score))                        # Get the scores when the episode ends
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores)))
                break

    return scores

scores = test()

fig = plt.figure()                              # Plotting the graph showing the increase in Average scores
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.plot(np.arange(len(scores)), scores)
plt.show()

env.close()                                     # Close the environment once you're done