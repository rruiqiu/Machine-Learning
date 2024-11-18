import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim


# seed = 8681
seed = 4003
np.random.seed(seed)
torch.manual_seed(seed)
transform = transforms.Compose([transforms.ToTensor(),  # Convert from PIL Image to tensors
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
train_dataset = datasets.FashionMNIST(root= './data', download=True, train=True, transform=transform)

train_size = int(0.8 * len(train_dataset))  # 80% for training
validation_size = len(train_dataset) - train_size  # Remaining 20% for validation
train_set, validation_set = random_split(train_dataset, [train_size, validation_size])


trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
validloader = DataLoader(validation_set, batch_size=128, shuffle=True)

# Download and load the test data
test_dataset = datasets.FashionMNIST(root= './data', download=True, train=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)




class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden_1 = nn.Linear(784, 156)
        self.hidden_2 = nn.Linear(156, 92)
        self.hidden_3 = nn.Linear(92, 80)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(80, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        # Output layer with softmax activation
        x = F.log_softmax(self.output(x), dim=-1)
        
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")       
model = Classifier()
# model.to(device)
criterion = nn.NLLLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 20

train_losses, test_losses = [], []
prev_train_accuracy = 0
for e in range(epochs):
    tot_train_loss = 0
    for images, labels in trainloader:
      # Flatten MNIST images into a 784 long vector
      images = images.view(images.shape[0], -1)
      # images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      
      output = model(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      
      tot_train_loss += loss.item()
      # print(f"Training loss: {tot_train_loss/len(trainloader)}")
    else:
      tot_test_loss = 0
      test_correct = 0  # Number of correct predictions on the test set
      # Turn off gradients for validation, saves memory and computations
      with torch.no_grad():
        for images, labels in validloader:
          images = images.view(images.shape[0], -1)
          # images, labels = images.to(device), labels.to(device)
          ps = model(images)
          loss = criterion(ps, labels)
          tot_test_loss += loss.item()

          ps = torch.exp(ps)
          top_p, top_class = ps.topk(1, dim=1)
          equals = top_class == labels.view(*top_class.shape)
          test_correct += equals.sum().item()
        
      train_loss = tot_train_loss / len(trainloader.dataset)
      test_loss = tot_test_loss / len(validloader.dataset)

      # At completion of epoch
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      train_accuracy = test_correct / len(validloader.dataset)
      
      print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(train_loss),
            "Test Loss: {:.3f}.. ".format(test_loss),
            "Train Accuracy: {:.3f}".format(test_correct / len(validloader.dataset)))


test_correct = 0      
for images, labels in testloader:
  images = images.view(images.shape[0], -1)
  # images, labels = images.to(device), labels.to(device)
  ps = model(images)
  loss = criterion(ps, labels)
  # tot_test_loss += loss.item()

  ps = torch.exp(ps)
  top_p, top_class = ps.topk(1, dim=1)
  equals = top_class == labels.view(*top_class.shape)
  test_correct += equals.sum().item()
  
actual_test_accuracy = test_correct/len((testloader.dataset))
print("Test Accuracy: {:.3f}".format(actual_test_accuracy))  
  
plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()