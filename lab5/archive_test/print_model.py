import pickle

def load_agent(file_name):
  """
  Load your agent's model from a file.

  Args:
      file_name (str): The file name to load the model from.
  """
  # TODO: Implement code to load your model
  # Example: for Q-learining:
  with open(file_name, 'rb') as f:
      data = pickle.load(f)
      q_table = data['q_table']
      iht = data['iht_dict']
  print(f"Model loaded from {file_name}.")
  print(q_table)
  # print(iht)
  
agent_model_file = r'C:\Users\Richard\Documents\4SL4\Machine-Learning\lab5\best_model.pkl'  # Set the model file name
load_agent(agent_model_file)