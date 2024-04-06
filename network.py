import numpy as np
import random
import json

from hyperparameters import hyperparameters
from dataset import train_data, test_data

# activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# loss function
def loss_function_prime(x, y):
    return 2 * (x - y)

# neural network
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        
        self.costs = []
        self.accuracies = []

        self.init_params() 

    def init_params(self):
        self.weights = [np.random.randn(j, i) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.randn(i, 1) for i in self.layers[1:]]

        self.momentum_w = [np.zeros((j, i)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.momentum_b = [np.zeros((i, 1)) for i in self.layers[1:]]

        self.velocity_w = [np.zeros((j, i)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.velocity_b = [np.zeros((i, 1)) for i in self.layers[1:]]
        
    def cost_function(self, data):
        cost = 0.0

        for x, y in train_data:
            _, _, a = self.feedforward(x)
            cost += np.linalg.norm(a - y) ** 2
        
        return (1 / ( 2 * len(data))) * cost

    def feedforward(self, x):
        pre_activations = []
        activations = [x]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, x) + b

            pre_activations.append(z)

            x = sigmoid(z)

            activations.append(x)

        return pre_activations, activations, x
    
    def backpropagation(self, pre_activations, activations, y):
        curr_dldw = [np.zeros(w.shape) for w in self.weights]
        curr_dldb = [np.zeros(b.shape) for b in self.biases]

        dldx = loss_function_prime(activations[-1], y) * sigmoid_prime(pre_activations[-1])
        curr_dldw[-1] = np.dot(dldx, activations[-2].transpose())
        curr_dldb[-1] = dldx

        for i in range(2, self.num_layers):
            dldx = np.dot(self.weights[-i + 1].transpose(), dldx) * sigmoid_prime(pre_activations[-i]) 
            curr_dldw[-i] = np.dot(dldx, activations[-i - 1].transpose())
            curr_dldb[-i] = dldx

        return curr_dldw, curr_dldb

    def SGD(self, train_data, mini_batch_size, lr, beta, gamma, epsilon):
        mini_batches = [train_data[j: j + mini_batch_size] for j in range(0, len(train_data), mini_batch_size)]

        for mini_batch in mini_batches:
            sum_dldw = [np.zeros(w.shape) for w in self.weights]
            sum_dldb = [np.zeros(b.shape) for b in self.biases]

            for x, y in mini_batch:
                pre_activations, activations, _ = self.feedforward(x)
                curr_dldw, curr_dldb = self.backpropagation(pre_activations, activations, y)

                sum_dldw = [a + b for a, b in zip(sum_dldw, curr_dldw)]
                sum_dldb = [a + b for a, b in zip(sum_dldb, curr_dldb)]
            
            self.momentum_w = [beta * mw + (1 - beta) * dw for mw, dw in zip(self.momentum_w, sum_dldw)]
            self.momentum_w = [beta * mb + (1 - beta) * db for mb, db in zip(self.momentum_w, sum_dldw)]

            self.velocity_w = [gamma * vw + (1 - gamma) * (dw ** 2) for vw, dw in zip(self.velocity_w, sum_dldw)]
            self.velocity_b = [gamma * vb + (1 - gamma) * (db ** 2) for vb, db in zip(self.velocity_b, sum_dldb)]

            self.weights = [w - lr * (mw / (np.sqrt(vw) + epsilon)) for w, mw, vw in zip(self.weights, self.momentum_w, self.velocity_w)]
            self.biases = [b - lr * (mb / (np.sqrt(vb) + epsilon)) for b, mb, vb in zip(self.biases, self.momentum_b, self.velocity_b)]
        
    def train(self, train_data, epochs, mini_batch_size, lr, beta, gamma, epsilon, test_data=None):
        for epoch in range(epochs):
            random.shuffle(train_data)

            cost = round(self.cost_function(train_data), 3)

            self.costs.append(cost)

            self.SGD(train_data, mini_batch_size, lr, beta, gamma, epsilon)

            if test_data:
                correct = self.test(test_data)
                accuracy = round((correct / len(test_data) * 100), 2)

                self.accuracies.append(accuracy)
                
            print(f'epoch {epoch + 1}/{epochs} | loss: {cost} accuracy: {accuracy}%')
        
        return accuracy

    def test(self, test_data):
        test_results = []

        for x, y in test_data:
            _, _, a = self.feedforward(x)

            test_results.append((np.argmax(a), y))

        return sum(int(x == y) for (x, y) in test_results)
    
    def save(self, filename):
        data = {"layers": self.layers, 
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                }
        
        with open(filename, 'w') as f:
            json.dump(data, f)

# create nn
input_size = len(train_data[0][0])

nn = NeuralNetwork([input_size, 128, 128, 128, 128, 10])

# train nn
epochs = hyperparameters['epochs']
mini_batch_size = hyperparameters['mini batch size']
lr = hyperparameters['learning rate']
beta = hyperparameters['momentum']
gamma = hyperparameters['gamma']
epsilon = hyperparameters['weight decay']

accuracy = nn.train(train_data, epochs, mini_batch_size, lr, beta, gamma, epsilon, test_data)

if accuracy >= 96:
    nn.save('parameters.json') # save parameters
