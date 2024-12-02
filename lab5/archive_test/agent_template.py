# agent_template.py

import gymnasium as gym
import numpy as np
from state_discretizer import StateDiscretizer
import pickle
class LunarLanderAgent:
    def __init__(self):
        """
        Initialize your agent here.

        This method is called when you create a new instance of your agent.
        Use this method to Initializes the environment, the agent’s model (e.g., Q-table or neural network),
        and the optional state discretizer for Q-learning. Add any necessary initialization for model parameters here.
        """
        # TODO: Initialize your agent's parameters and variables

        # Initialize environment
        # self.env = gym.make('LunarLander-v3',render_mode="human")
        self.env = gym.make('LunarLander-v3')
        self.num_actions = self.env.action_space.n # Action space size

        # # Initialize state discretizer if you are going to use Q-Learning
        self.state_discretizer = StateDiscretizer(self.env)

        # initialize Q-table or neural network weights
        self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.num_actions)]

        # Set learning parameters
        self.alpha = 0.1
        # self.alpha = alpha / self.state_discretizer.num_tilings  # Learning rate per tiling
        self.epsilon =  1.0       # Initial exploration rate
        self.epsilon_decay = 0.995   # Exploration decay rate
        # ..

        # Initialize any other parameters and variables
        # ...
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.gamma = 0.95  # Discount factor
        
        pass

    def select_action(self, state):
        """
        Given a state, select an action to take. The function should operate in training and testing modes,
        where in testing you will need to shut off epsilon-greedy selection.

        Args:
            state (array): The current state of the environment.

        Returns:
            int: The action to take.
        """
        # TODO: Implement your action selection policy here
        # For example, you might use an epsilon-greedy policy if you're using Q-learning
        # Ensure the action returned is an integer in the range [0, 3]
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore 
        else:
            state_features = self.state_discretizer.discretize(state)
            q_values = [np.sum(self.q_table[a][state_features]) for a in range(self.num_actions)]
            return np.argmax(q_values)  # Exploit
        # Discretize the state if you are going to use Q-Learning
        # state_features = self.state_discretizer.discretize(state)


    def train(self, num_episodes):
        """
         Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        # TODO: Implement your training loop here
        # Make sure to:
        # 1) Evaluate the training in each episode by monitoring the average of the previous ~100
        rewards = []
        max_reward = -float('inf')  # Initialize to negative infinity to track the best reward
        for episode in range(num_episodes):
        # while max_reward<100:
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
        
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.gamma = max(0.95, self.gamma * 0.995)  # Gradually reduce gamma to stabilize learning

            # print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            rewards.append(total_reward)
            if total_reward > max_reward:
                self.save_agent('model.pkl')
                print("Replace Max_reward is",max_reward,"new reward",total_reward)
                print("epsilon is",self.epsilon)
                max_reward = total_reward
                
        #    episodes cumulative rewards (return).
        # 2) Autosave the best model achived in each epoch based on the evaluation.
        print("Max_reward is",max_reward)
        # print("Total rewards arr",rewards)

    def update(self, state, action, reward, next_state, done):
        """
        Update your agent's knowledge based on the transition.

        Args:
            state (array): The previous state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The new state after the action.
            done (bool): Whether the episode has ended.
        """
        # TODO: Implement your agent's update logic here
        # This method is where you would update your Q-table or neural network

        # Discretize the states if you are going to use Q-Learning
        state_features = self.state_discretizer.discretize(state)
        next_state_features = self.state_discretizer.discretize(next_state)
        if done:
            target = reward
        else:
            max_next_q = max([np.sum(self.q_table[a][next_state_features]) for a in range(self.num_actions)]) #?
            target = reward + self.gamma * max_next_q
        
        try:
            self.q_table[action][state_features] += self.alpha * (target - self.q_table[action][state_features])
        except:
            print("debug")
            
    
    def test(self, num_episodes = 100):
        """
        Test your agent locally before submission to get a hint of the expected score.

        Args:
            num_episodes (int): Number of episodes to test for.
        """
        # TODO: Implement your testing loop here
        # Make sure to:
        # Store the cumulative rewards (return) in all episodes and then take the average
        
        total_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_features = self.state_discretizer.discretize(state)
                q_values = [np.sum(self.q_table[a][state_features]) for a in range(self.num_actions)]
                action = np.argmax(q_values)  # Always exploit during testing
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)

        average_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {average_reward}")
        return average_reward        


    def save_agent(self, file_name):
        """
        Save your agent's model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        # TODO: Implement code to save your model (e.g., Q-table, neural network weights)
        # Example: for Q-learining:
        with open(file_name, 'wb') as f:
          pickle.dump({
              'q_table': self.q_table,
              'iht_dict': self.state_discretizer.iht.dictionary,
            #   'max_reward': self.max_reward
          }, f)
        print(f"Agent saved to {file_name}.")

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        # TODO: Implement code to load your model
        # Example: for Q-learining:
        with open(file_name, 'rb') as f:
           data = pickle.load(f)
           self.q_table = data['q_table']
           self.state_discretizer.iht.dictionary = data['iht_dict']
        print(f"Model loaded from {file_name}.")


if __name__ == '__main__':

    agent = LunarLanderAgent()
    agent_model_file = r'C:\Users\Richard\Documents\4SL4\Machine-Learning\lab5\model.pkl'  # Set the model file name

    # try:
    #     agent.load_agent(agent_model_file)
    #     agent.epsilon = 0.5  # Set to desired starting exploration rate
    #     agent.epsilon_decay = 0.99  # Adjust decay rate as needed
    #     print("Loaded pre-trained agent.")
    # except FileNotFoundError:
    #     print(f"No saved model found at {agent_model_file}. Starting from scratch.")

    
    # # Example usage:
    # # Uncomment the following lines to train your agent and save the model

    # num_training_episodes = 1000  # Define the number of training episodes
    # print("Training the agent...")
    # agent.train(num_training_episodes)
    # print("Training completed.")

    # Save the trained model
    print("Tesing the agent")
    agent.test()