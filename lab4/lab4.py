import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(3093)

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss Function
def cross_entropy_loss(y_pred, y_true):
    n_samples = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples
    return loss

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, weight_decay=0.001):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def forward(self, X):
        self.activations = []
        self.z_values = []

        a = X
        self.activations.append(a)

        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = relu(z)
            self.activations.append(a)

        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = softmax(z)
        self.activations.append(a)

        return a

    def backward(self, X, y):
        m = y.shape[0]
        deltas = [self.activations[-1] - y]

        for i in range(len(self.weights) - 1, 0, -1):
            deltas.append(np.dot(deltas[-1], self.weights[i].T) * relu_derivative(self.z_values[i - 1]))

        deltas.reverse()

        for i in range(len(self.weights)):
            dw = np.dot(self.activations[i].T, deltas[i]) / m + self.weight_decay * self.weights[i]
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        train_loss = []
        val_loss = []

        for epoch in range(epochs):
            perm = np.random.permutation(X_train.shape[0])
            X_train = X_train[perm]
            y_train = y_train[perm]

            for batch in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[batch:batch + batch_size]
                y_batch = y_train[batch:batch + batch_size]

                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            train_loss.append(cross_entropy_loss(self.forward(X_train), y_train))
            val_loss.append(cross_entropy_loss(self.forward(X_val), y_val))
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

        return train_loss, val_loss

# Data Preprocessing (refer to the assignment document)
from torchvision import datasets

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True)

X_train = train_dataset.data.numpy().reshape(-1, 28 * 28).astype('float32') / 255.0
y_train = np.eye(10)[train_dataset.targets.numpy()]

X_test = test_dataset.data.numpy().reshape(-1, 28 * 28).astype('float32') / 255.0
y_test = np.eye(10)[test_dataset.targets.numpy()]

val_size = int(0.2 * X_train.shape[0])
X_val, y_val = X_train[:val_size], y_train[:val_size]
X_train, y_train = X_train[val_size:], y_train[val_size:]

# Training the Network
nn = NeuralNetwork(input_size=784, hidden_layers=[156, 92], output_size=10, learning_rate=0.01, weight_decay=0.001)
train_loss, val_loss = nn.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)

# Plot Learning Curves
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.title('Learning Curves')
plt.show()
