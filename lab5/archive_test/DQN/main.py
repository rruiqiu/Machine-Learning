import gymnasium as gym
import time
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent_DQN import Agent


def dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,goal=300.0,max_t=1000):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    try:
      for i_episode in range(1, n_episodes+1):
        state,info = env.reset()
        score = 0
        # done = False
        for t in range(max_t):
          action = agent.select_action(state, eps)
          next_state, reward, done, info,_ = env.step(action)
          agent.step(state, action, reward, next_state, done)
          state = next_state
          score += reward
          if done:
              break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        # print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.8f}'.format(i_episode, np.mean(scores_window), eps), end="")
        print('Episode {}\tAverage Score: {:.2f}\tEpsilon: {:.8f}'.format(i_episode, np.mean(scores_window), eps))
        # if i_episode % 100 == 0:
        #   print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.8f}'.format(i_episode, np.mean(scores_window), eps))
        if np.mean(scores_window)>=goal:
          print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tEpsilon: {:.8f}'.format(i_episode-100, np.mean(scores_window), eps))
          torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_lunar_'+str(goal)+'.pth')
          break
          
    except KeyboardInterrupt:
      print("\nTraining interrupted. Saving the current model...")
      torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_lunar_interrupted.pth')

    return scores

# scores = dqn()

def test_dqn(filename,num_episodes=100):
  total_rewards = []
  env = gym.make('LunarLander-v3')
  agent.qnetwork_local.load_state_dict(torch.load(filename, weights_only=True))

  for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)  # Exploit only
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    total_rewards.append(episode_reward)
    # print(f"Test Episode {episode + 1}, Score: {episode_reward}")
  avg_reward = np.mean(total_rewards)
  print(f"Average reward over {num_episodes} episodes: {avg_reward:.4f}")
  return avg_reward    

env = gym.make('LunarLander-v3')
agent = Agent(state_size=8, action_size=4, seed=0)
agent.load_agent("checkpoint_lunar_288.pth")  # Load from your saved model file

n_episodes=2000

eps_start=0.035
eps_end=0.023
eps_decay=0.9995
goal=260.0
max_t=1000
scores = dqn(n_episodes, eps_start, eps_end, eps_decay,goal,max_t)


# filename = r'C:\Users\Richard\Documents\4SL4\Machine-Learning\lab5\DQN\checkpoint_lunar_288.pth'
# test_dqn(filename)

# 0.0280

# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()