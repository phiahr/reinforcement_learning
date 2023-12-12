# Reinforcement Learning Problem 1
# Oscar Eriksson, 0011301991, oscer@kth.se
# Philip Ahrendt, 960605R119, pcah@kth.se
# Load packages
import numpy as np
import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU()
        )
        self.mu = nn.Sequential(
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, output_size),
            nn.Tanh()
        )
        self.cov = nn.Sequential(
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        state = self.input_layer(state)
        return self.mu(state), self.cov(state)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, cov = self.forward(state)
        mu = mu.detach().numpy()
        cov = cov.detach().numpy()
        return np.clip(np.random.normal(mu, np.sqrt(cov)), -1, 1)

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
    
    def forward(self, state):
        return self.critic(state)