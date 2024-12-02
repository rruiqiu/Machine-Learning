import gymnasium as gym
import numpy as np
import random
import pickle
from collections import deque
from state_discretizer import StateDiscretizer

class LunarLanderAgent:
    def __init__(self):
        """
        Initialize your agent here.
        """
        # Initialize environment
        self.env = gym.make('LunarLander-v3')

        # Initialize state discretizer
        self.state_discretizer = StateDiscretizer(self.env)

        # Define action size
        self.num_actions = self.env.action_space.n  # Should be 4

        # Initialize Q-table
        # Each action has an array of weights corresponding to the IHT size
        self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.num_actions)]

        # Hyperparameters
        self.alpha = 0.1 / self.state_discretizer.num_tilings  # Learning rate per tiling
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.88

        self.training_mode = True  # Flag to switch between training and testing

        self.steps_done = 0  # For epsilon decay

    def select_action(self, state):
        """
        Given a state, select an action to take.
        """
        state_indices = self.state_discretizer.discretize(state)

        if not self.training_mode or random.random() > self.epsilon:
            # Exploit: choose the action with the highest Q-value
            q_values = [np.sum(self.q_table[a][state_indices]) for a in range(self.num_actions)]
            action = np.argmax(q_values)
        else:
            # Explore: choose a random action
            action = random.randrange(self.num_actions)
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
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            scores.append(total_reward)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode}, Score: {total_reward}, Avg Score: {avg_score:.2f}, Epsilon: {self.epsilon:.2f}")
            # Save the model if it achieves a new best average score
            if avg_score > best_avg_score and self.epsilon <= self.epsilon_min:
                best_avg_score = avg_score
                self.save_agent('best_model.pkl')
        # Optionally plot avg_scores

    def update(self, state, action, reward, next_state, done):
        """
        Update your agent's knowledge based on the transition.
        """
        # Discretize the states
        state_indices = self.state_discretizer.discretize(state)
        next_state_indices = self.state_discretizer.discretize(next_state)

        # Compute the current Q-value
        q_current = np.sum(self.q_table[action][state_indices])

        # Compute the maximum Q-value for the next state
        q_next = max([np.sum(self.q_table[a][next_state_indices]) for a in range(self.num_actions)])

        # Compute the target Q-value
        target = reward + (0 if done else self.gamma * q_next)

        # Compute the TD error
        td_error = target - q_current

        # Update the Q-values for each active tile
        for idx in state_indices:
            self.q_table[action][idx] += self.alpha * td_error

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
        with open(file_name, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'iht_dict': self.state_discretizer.iht.dictionary
            }, f)
        print(f"Model saved to {file_name}.")

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.
        """
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.state_discretizer.iht.dictionary = data['iht_dict']
        self.training_mode = False  # Disable training mode when loading the agent
        print(f"Model loaded from {file_name}.")

if __name__ == '__main__':

    agent = LunarLanderAgent()
    agent_model_file = 'model.pkl'  # Set the model file name

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
