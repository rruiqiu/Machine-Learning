import gymnasium as gym
import numpy as np
from state_discretizer import StateDiscretizer
from collections import deque
import pickle

class LunarLanderAgent:
    def __init__(self):
        self.env = gym.make('LunarLander-v3')
        self.num_tilings=32
        self.tiles_per_dim=8
        self.iht_size=4096
        self.state_discretizer = StateDiscretizer(self.env,self.num_tilings,self.tiles_per_dim,self.iht_size)
        self.state_discretizer = StateDiscretizer(self.env)
        # Q-learning parameters
        self.num_actions = self.env.action_space.n
        # self.alpha = 0.1 / self.state_discretizer.num_tilings
        self.alpha = 0.1 / self.state_discretizer.num_tilings
        self.gamma = 0.99 #discount factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01

        # Initialize Q-table
        self.q_table = [np.zeros(self.state_discretizer.iht_size)
                        for _ in range(self.num_actions)]

        # Initialize performance tracking
        self.performance = deque(maxlen=100)
        self.best_performance = -np.inf

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            state_features = self.state_discretizer.discretize(state)
            q_values = [np.sum(self.q_table[a][state_features])
                        for a in range(self.num_actions)]
            return np.argmax(q_values)  # Exploit

    def train(self, num_episodes):
        for episode in range(num_episodes):
            
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

            self.performance.append(episode_reward)
            avg_performance = np.mean(self.performance)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            # self.gamma = max(0.95, self.gamma * 0.995)
            
            if avg_performance > self.best_performance and self.epsilon == self.epsilon_min:
                self.best_performance = avg_performance
                self.save_agent("best_model.pkl")
                print(f"Best model saved with average reward: {self.best_performance}")

            print(f"Episode: {episode + 1}/{num_episodes}, "
                  f"Epsilon: {self.epsilon:.3f}, "
                  f"Average Reward: {avg_performance:.2f}")

    def update(self, state, action, reward, next_state, done):
        state_features = self.state_discretizer.discretize(state)
        next_state_features = self.state_discretizer.discretize(next_state)

        if done:
            target = reward
        else:
            q_values_next = [np.sum(self.q_table[a][next_state_features])
                             for a in range(self.num_actions)]
            target = reward + self.gamma * np.max(q_values_next)

        # Update Q-table
        self.q_table[action][state_features] += self.alpha * (
            target -np.sum(self.q_table[action][state_features])
        )

    def test(self, num_episodes=100):
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state)  # Exploit only
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward

    def save_agent(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'iht_dict': self.state_discretizer.iht.dictionary
            }, f)

    def load_agent(self, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.state_discretizer.iht.dictionary = data['iht_dict']
        print(f"Model loaded from {file_name}.")

if __name__ == '__main__':
    # agent = LunarLanderAgent()
    # agent_model_file = 'model.pkl'

    # num_training_episodes = 1000
    # print("Training the agent...")
    # agent.train(num_training_episodes)
    # print("Training completed.")
    # print("Best training Performance: ",agent.best_performance)
    # agent.save_agent(agent_model_file)
    # print("Model saved.")
    
    #test the model
    agent = LunarLanderAgent()
    agent_model_file = 'best_model_157.pkl'
    agent.load_agent(agent_model_file)
    agent.epsilon = 0
    agent.test()