# Reinforcement Learning Lab 2
# Oscar Eriksson, 0011301991, oscer@kth.se
# Philip Ahrendt, 960605R119, pcah@kth.se
# Load packages
import numpy as np
import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, output_size),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.actor(state)
    
    def act(self, state, noise):
        state = torch.from_numpy(state).float().unsqueeze(0)
        return np.clip(self.forward(state).cpu().detach().numpy() + noise, -1, 1)

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size + output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        stack = torch.hstack([state, action])
        return self.critic(stack)