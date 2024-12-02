# agent_template.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque




class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Define the neural network architecture
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class LunarLanderAgent:
    def __init__(self):
        """
        Initialize your agent here.
        """
        # Initialize environment
        self.env = gym.make('LunarLander-v3')

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define state and action sizes
        self.state_size = self.env.observation_space.shape[0]  # Should be 8
        self.action_size = self.env.action_space.n  # Should be 4

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.target_update = 10  # How often to update the target network

        # Experience replay buffer
        self.memory = deque(maxlen=100000)

        # Q-Network
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.steps_done = 0  # For epsilon decay
        self.training_mode = True  # Flag to switch between training and testing

    def select_action(self, state):
        """
        Given a state, select an action to take.
        """
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        if not self.training_mode or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        else:
            action = random.randrange(self.action_size)
        return action

    def train(self, num_episodes):
        """
        Contains the main training loop where the agent learns over multiple episodes.
        """
        self.training_mode = True  # Enable training mode
        scores = []
        avg_scores = []
        best_avg_score = -float('inf')
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                self.update()
            scores.append(total_reward)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Episode {episode}, Score: {total_reward}, Avg Score: {avg_score:.2f}, Epsilon: {self.epsilon:.2f}")
            # Save the model if it achieves a new best average score
            if avg_score > best_avg_score and self.epsilon <= self.epsilon_min:
                best_avg_score = avg_score
                self.save_agent('best_model.pth')
        # Optionally plot avg_scores

    def update(self):
        """
        Update your agent's knowledge based on the transition.
        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute Q values
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, num_episodes=100):
        """
        Test your agent locally before submission to get a hint of the expected score.
        """
        self.training_mode = False  # Disable training mode
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}, Score: {total_reward}")
        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    def save_agent(self, file_name):
        """
        Save your agent's model to a file.
        """
        torch.save(self.policy_net.state_dict(), file_name)
        print(f"Model saved to {file_name}.")

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.
        """
        self.policy_net.load_state_dict(torch.load(file_name, map_location=self.device))
        self.policy_net.eval()
        # Also update target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.training_mode = False  # Disable training mode when loading the agent
        print(f"Model loaded from {file_name}.")

if __name__ == '__main__':

    agent = LunarLanderAgent()
    agent_model_file = 'model.pth'  # Set the model file name

    # Uncomment the following lines to train your agent and save the model

    num_training_episodes = 1000  # Define the number of training episodes
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")

    # Save the trained model
    agent.save_agent(agent_model_file)
    print("Model saved.")

    # Test the agent
    print("Testing the agent...")
    agent.test(num_episodes=100)