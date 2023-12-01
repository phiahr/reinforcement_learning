import torch 
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, num_inputs, num_hidden_neurons, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden_neurons) 
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden_neurons, num_outputs)

    # In the forward function you define how each layer is
    # interconnected. Observe that 'x' is the input.
    def forward(self, x):
        # First layer (input layer)
        x = self.linear1(x)
        x = self.act1(x)

        # Second layer (output)
        x = self.linear2(x)
        return x
    
    def loss(self, x, y):
        return np.sum((x-y)**2)
        # return nn.MSELoss()(x, y)