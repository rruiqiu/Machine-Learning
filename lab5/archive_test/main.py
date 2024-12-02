import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import gymnasium as gym
# ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#q-network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
      
      
Transition = namedtuple('Transition',('state', 'action', 'next_state','reward'))

class ReplayMemory(object):
    def __init__(self, capacity,device):
        self.memory = deque([], maxlen=capacity)
        self.device = device
    def push(self, state, action, next_state, reward):
        """Save a transition"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device) if next_state is not None else None
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
      

class LunarLanderAgent():
  def __init__(self):
    
    #init the env
    self.env = gym.make("LunarLander-v3")  
    
    #init the Q-learning
    self.alpha = 5e-4
    self.gamma = 0.99
    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.01
    
    # Q-Network
    self.state_size = 8
    self.action_size = 4
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Q-Network
    self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
    self.target_net = DQN(self.state_size, self.action_size).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    # self.target_net.eval()
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
    # self.target_net.eval()
    #Replace Memory
    self.batch_size = 64
    self.memory = ReplayMemory(int(1e5),self.device)
    # self.steps_done =0
    
    
    #others
    self.tau = 1e-3 #for soft updates
    self.performance = deque(maxlen=100)
    self.best_performance = -np.inf
    
  def select_action(self,state):
    state = torch.tensor(state, dtype=torch.float32).to(self.device)
    if random.random() > self.epsilon:
      with torch.no_grad():
        q_values = self.policy_net(state)
        action = q_values.argmax().item()
        return action
    else:
      action = random.randrange(self.action_size)
      return action
  
  def update(self):
    if len(self.memory) < self.batch_size:
      return
    transitions = self.memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(self.device)  # Shape: [batch_size, state_dim]
    action_batch = torch.cat(batch.action).unsqueeze(1).to(self.device)  # Shape: [batch_size, 1]
    reward_batch = torch.cat(batch.reward).to(self.device)  # Shape: [batch_size, 1]

    
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(self.batch_size, device=self.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()    
    
    
    
  def train(self,goal,num_episodes=1000):
    try:
      for episode in range(num_episodes):
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        done = False
        total_rewards = 0
        while not done:
          action = self.select_action(state)
          next_state, reward, terminated, truncated, _ = self.env.step(action)
          done = terminated or truncated
          if terminated:
              next_state = None
          else:
              next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
          self.memory.push(state,action,next_state,reward)
          total_rewards += reward
          state = next_state
          self.update()
          
          target_net_state_dict = self.target_net.state_dict()
          policy_net_state_dict = self.policy_net.state_dict()
          for key in policy_net_state_dict:
              target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
          self.target_net.load_state_dict(target_net_state_dict)
          
        self.performance.append(total_rewards)
        avg_performance = np.mean(self.performance)
        self.epsilon = max(self.epsilon_min,self.epsilon * self.epsilon_decay)
        
        print(f"Episode: {episode + 1}/{num_episodes}, "
              f"Epsilon: {self.epsilon:.3f}, "
              f"Average Reward: {avg_performance:.2f}")


        if(avg_performance >= goal and avg_performance > self.best_performance and len(self.performance)==100):
          self.best_performance = avg_performance
          torch.save(self.policy_net.state_dict(), 'best_model.pth')
          # break
    except KeyboardInterrupt:
      print("\nTraining interrupted. Saving the current model...")
      torch.save(self.policy_net.state_dict(), 'model_interrupted.pth')
    
  def load_agent(self, file_name):
      """Loads a saved model."""
      self.policy_net.load_state_dict(torch.load(file_name))
      self.target_net.load_state_dict(torch.load(file_name))
      print(f"Model loaded from {file_name}")
      
  def test(self, num_episodes=100):
      total_rewards = []
      # agent.qnetwork_local.load_state_dict(torch.load(filename, weights_only=True))

      for episode in range(num_episodes):
          state, _ = self.env.reset()
          done = False
          episode_reward = 0

          while not done:
              action = self.select_action(state)  # Exploit only
              next_state, reward, terminated, truncated, _ = self.env.step(action)
              state = next_state
              episode_reward += reward
              done = terminated or truncated
              
          total_rewards.append(episode_reward)

      avg_reward = np.mean(total_rewards)
      print(f"Test Average reward over {num_episodes} episodes: {avg_reward:.4f}")
      return avg_reward    
if __name__ == '__main__':

    agent = LunarLanderAgent()       # Initialize the agent
    agent.load_agent('best_model.pth')
    agent.epsilon = 0.03
    agent.epsilon_min = 0.01
    agent.epsilon_decay = 0.995
    agent.train(200.0)
    
    # agent.load_agent('best_model.pth')
    # agent.epsilon = 0.0
    # print("test average",agent.test())