# Reinforcement Learning Problem 1
# Oscar Eriksson, 0011301991, oscer@kth.se
# Philip Ahrendt, 960605R119, pcah@kth.se
# Load packages
import numpy as np
import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size),
        )

    def forward(self, x):
        return self.fc(x)
    
    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(4)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_value = self.forward(state)
            return torch.argmax(q_value).item()