from torchvision import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
seed = 8681
# seed = 4003
np.random.seed(seed)
torch.manual_seed(seed)


def init_data():
  train_dataset = datasets.FashionMNIST(root= './data', train=True,download=True)
  test_dataset = datasets.FashionMNIST(root= './data', train=False,download=True)

  print(train_dataset.data.shape, test_dataset.data.shape)


  num_classes = 10
  X_train = train_dataset.data.numpy().reshape(-1, 28 * 28).astype(
  'float32' ) / 255.0
  Y_train = train_dataset.targets.numpy()

  X_test = test_dataset.data.numpy().reshape(-1, 28 * 28).astype(
  'float32' ) / 255.0
  Y_test = test_dataset.targets.numpy()


  # Split the training set into train and validation sets (80% /20%)
  validation_size = int(0.2 * X_train.shape[0])
  X_validation, Y_validation = X_train[:validation_size], Y_train[:validation_size]
  X_train, Y_train = X_train[validation_size:], Y_train[validation_size:]

  # Save original labels before one-hot encoding
  Y_train_orig = Y_train
  Y_validation_orig = Y_validation
  Y_test_orig = Y_test
  # Convert labels to one-hot encoding for multi-class classification
  def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

  Y_train = one_hot_encode(Y_train, num_classes)
  Y_validation = one_hot_encode(Y_validation, num_classes)
  Y_test = one_hot_encode(Y_test, num_classes)
  # Calculate the mean and standard deviation of the training features
  X_train_mean = X_train.mean(axis=0)
  X_train_std = X_train.std(axis=0)
  X_train_std[X_train_std == 0] = 1  # To avoid division by zero
  # Standardize all three subsets of data
  X_train = (X_train - X_train_mean) / X_train_std
  X_validation = (X_validation - X_train_mean) / X_train_std
  X_test = (X_test - X_train_mean) / X_train_std
  return X_train,X_validation,X_test,Y_train,Y_validation,Y_test

def create_batches(X_train, Y_train, batch_size=128):
  # Generate a list of indices and shuffle them
  indices = np.arange(X_train.shape[0])
  np.random.shuffle(indices)

  # Apply the shuffled indices to the data
  X_train = X_train[indices]
  Y_train = Y_train[indices]

  # Generate batches
  for start_idx in range(0, X_train.shape[0], batch_size):
    end_idx = start_idx + batch_size
    yield X_train[start_idx:end_idx], Y_train[start_idx:end_idx]

class Classifier():
  def __init__(self):
    # np.random.seed()
    self.w_1 = np.random.rand(784, 156) * 0.01
    self.b_1 = np.zeros((1, 156))
    
    self.w_2 = np.random.rand(156, 92) * 0.01  
    self.b_2 = np.zeros((1, 92))
    
    self.w_3 = np.random.rand(92, 80) * 0.01  
    self.b_3 = np.zeros((1, 80))
    
    self.w_4 = np.random.rand(80, 10) * 0.01  
    self.b_4 = np.zeros((1, 10))

    # self.w_1 = np.random.randn(784, 156) * np.sqrt(2 / (784 + 156))
    # self.w_2 = np.random.randn(156, 92) * np.sqrt(2 / (156 + 92))
    # self.w_3 = np.random.randn(92, 80) * np.sqrt(2 / (92 + 80))
    # self.w_4 = np.random.randn(80, 10) * np.sqrt(2 / (80 + 10))
  def forward(self, x):
    # Input to first hidden layer
    self.z1 = np.dot(x, self.w_1) + self.b_1
    self.h1 = self.relu(self.z1)
    
    # First hidden layer to second hidden layer
    self.z2 = np.dot(self.h1, self.w_2) + self.b_2
    self.h2 = self.relu(self.z2)
    
    # Second hidden layer to third hidden layer
    self.z3 = np.dot(self.h2, self.w_3) + self.b_3
    self.h3 = self.relu(self.z3)
    
    # Third hidden layer to output layer
    self.z4 = np.dot(self.h3, self.w_4) + self.b_4
    self.h4 = self.softmax(self.z4)
    
    return self.h4    
    
  def backward(self, x, y_true, y_pred, learning_rate=0.01):
    # Calculate the gradient of the loss with respect to output layer
    #https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
    batch_size = x.shape[0]
    weight_decay_2 = 0.0018738
    weight_decay_3 = 0.00231
    weight_decay_4 = 0.001232
    d_loss_z4 = y_pred - y_true  # Derivative of cross-entropy loss w.r.t softmax output
    # Gradients for the output layer
    d_loss_w4 = np.dot(self.h3.T, d_loss_z4)/ batch_size + weight_decay_4 * self.w_4
    d_loss_b4 = np.sum(d_loss_z4, axis=0, keepdims=True)
    
    # Gradients for third hidden layer
    # test_relu = self.relu_derivative(self.z3)
    d_loss_z3 = self.relu_derivative(self.z3) * np.dot(d_loss_z4,self.w_4.T)
    d_loss_w3 = np.dot(self.h2.T, d_loss_z3)/ batch_size + weight_decay_3 * self.w_3
    d_loss_b3 = np.sum(d_loss_z3, axis=0, keepdims=True)
    
    # Gradients for second hidden layer
    d_loss_z2 = self.relu_derivative(self.z2) * np.dot(d_loss_z3,self.w_3.T)
    d_loss_w2 = np.dot(self.h1.T, d_loss_z2)/ batch_size + weight_decay_2 * self.w_2
    d_loss_b2 = np.sum(d_loss_z2, axis=0, keepdims=True)
    
    # Gradients for first hidden layer
    d_loss_z1 = self.relu_derivative(self.z1) * np.dot(d_loss_z2,self.w_2.T)
    d_loss_w1 = np.dot(x.T, d_loss_z1)/ batch_size
    d_loss_b1 = np.sum(d_loss_z1, axis=0, keepdims=True)
    
    # Update weights and biases using gradient descent
    self.w_4 -= learning_rate * d_loss_w4
    self.b_4 -= learning_rate * d_loss_b4
    
    self.w_3 -= learning_rate * d_loss_w3
    self.b_3 -= learning_rate * d_loss_b3
    
    self.w_2 -= learning_rate * d_loss_w2
    self.b_2 -= learning_rate * d_loss_b2
    
    self.w_1 -= learning_rate * d_loss_w1
    self.b_1 -= learning_rate * d_loss_b1
    
  def relu(self,x):
    return np.maximum(0, x)
  
  def softmax(self,x):
      exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
      return exp_x / np.sum(exp_x, axis=1, keepdims=True)  

  def relu_derivative(self, x):
    #values greater than 0 will return 1, ref: https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
    return (x > 0).astype(float)

def cross_entropy_loss(y_true,y_pred):
  loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
  return loss

# Example implementation
def compute_accuracy_with_numpy(ps, encoding_labels):
  # Get the index of the maximum probability (equivalent to top_class)
  top_class = np.argmax(ps, axis=1)  # Shape: [N]
  labels = np.argmax(encoding_labels, axis=1)
  # Compare predicted class indices with true labels
  equals = (top_class == labels)  # Shape: [N]

  # Sum up the number of correct predictions
  test_correct = np.sum(equals)

  return test_correct


def main():
  X_train,X_validation,X_test,Y_train,Y_validation,Y_test = init_data()
  # print(X_train[:10])
  
  classifier = Classifier()

  epochs = 0
  learning_rate = 0.01
  train_losses, test_losses = [], []
  # for epoch in range(epochs):
  
  train_accuracy = 0
  prev_train_accuracy = 0
  
  while 1:
    total_loss = 0
    for images,labels in create_batches(X_train,Y_train):
      # print(images,labels)
      output = classifier.forward(images)
      # print(output[0].sum())
      loss = cross_entropy_loss(labels,output)
      total_loss += loss
      # print(cross_entropy_loss)
      classifier.backward(images, labels, output, learning_rate)

    else:
      tot_test_loss = 0
      test_correct = 0
      for images, labels in create_batches(X_validation,Y_validation):
        # images = images.view(images.shape[0], -1)
        # images, labels = images.to(device), labels.to(device)
        output = classifier.forward(images)
        loss = cross_entropy_loss(labels,output)
        tot_test_loss += loss
        test_correct += compute_accuracy_with_numpy(output,labels)
      
      train_loss = total_loss / len(X_train)
      test_loss = tot_test_loss / len(X_validation)
      epochs = epochs + 1
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      train_accuracy = test_correct / len(X_validation)
      if(train_accuracy > 0.86 and prev_train_accuracy > train_accuracy):
        print("Early Stopping!",
            "Epoch: {}.. ".format(epochs),
            "Training Loss: {:.10f}.. ".format(train_loss),
            "Validation Loss: {:.10f}.. ".format(test_loss),
            "Train Accuracy: {:.3f}".format(train_accuracy))
        break
      prev_train_accuracy = train_accuracy
      print("Epoch: {}.. ".format(epochs),
            "Training Loss: {:.10f}.. ".format(train_loss),
            "Validation Loss: {:.10f}.. ".format(test_loss),
            "Train Accuracy: {:.3f}".format(train_accuracy))
    
  
  # for 
  test_accuracy = 0
  test_correct = 0
  for images, labels in create_batches(X_test,Y_test):
    output = classifier.forward(images)
    # loss = cross_entropy_loss(labels,output)
    # tot_test_loss += loss
    test_correct += compute_accuracy_with_numpy(output,labels)
  test_accuracy = test_correct / len(X_test)
  
  
  print("Test Accuracy: {:.3f}".format(test_accuracy))
  plt.figure()  
  plt.plot(train_losses, label='Training loss')
  plt.plot(test_losses, label='Validation loss')
  plt.legend(frameon=False)
  plt.title("Losses Over Epochs")
  
  
  plt.show()

  
  
if __name__ == "__main__":
  main()