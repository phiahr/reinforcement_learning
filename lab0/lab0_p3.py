# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 0 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 17th October 2020, by alessior@kth.se

### IMPORT PACKAGES ###
# numpy for numerical/random operations
# gym for the Reinforcement Learning environment
import numpy as np
import gym
from collections import deque
import NeuralNetwork as nn
import torch

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')


### CREATE RL ENVIRONMENT ###
# env = gym.make('CartPole-v0')        # Create a CartPole environment
env = gym.make('CartPole-v1', render_mode="human")        # Create a CartPole environment

n = len(env.observation_space.low)   # State space dimensionality
m = env.action_space.n               # Number of actions
num_hidden_layers = 8                # Number of hidden layers in the NN

buffer = deque(maxlen=100)

nn = nn.NeuralNet(n, num_hidden_layers, m) # Create the neural network

def sample_batch(buffer, batch_size):
    """
    Sample a batch of transitions from the replay buffer.

    :param buffer: Replay buffer
    :param batch_size: Batch size
    :return: A list of transitions
    """
    batch = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[i] for i in batch]

### PLAY ENVIRONMENT ###
# The next while loop plays 5 episode of the environment
for episode in range(5):
    state = env.reset()                  # Reset environment, returns initial
    state = state[0]
                                         # state
    done = False                         # Boolean variable used to indicate if
                                         # an episode terminated

    while not done:
        env.render()                     # Render the environment
                                         # (DO NOT USE during training of the
                                         # labs...)
        # action  = np.random.randint(m)   # Pick a random integer between
                                         # [0, m-1]

        state_tensor=torch.tensor([state], requires_grad=False)
        values = nn(state_tensor)
        action = values.max(1)[1].item()
        
        # The next line takes permits you to take an action in the RL environment
        # env.step(action) returns 4 variables:
        # (1) next state; (2) reward; (3) done variable; (4) additional stuff
        next_state, reward, done, _, _ = env.step(action)

        # append (s, a, r, s′, d) to the buffer
        buffer.append((state, action, reward, next_state, done))


        ### TRAINING ###
        optim = torch.optim.Adam(nn.parameters(), lr=0.01)
        # loss = nn.loss(state_tensor, action)

        # train if more than 3 samples
        if len(buffer) >= 3:
            samples = sample_batch(buffer, 3)
            optim.zero_grad()
            z = []
            for sample in samples:
                state, action, reward, next_state, done = sample
                state_tensor = torch.tensor([state], requires_grad=True)
                action = nn(state_tensor).max().item()
                z.append(action)
            
            z = torch.tensor(z, requires_grad=True)
            y = torch.zeros(3, requires_grad=True)

            loss=torch.nn.functional.mse_loss(z,y)
            loss.backward()
            optim.step()

            torch.nn.utils.clip_grad_norm(nn.parameters(), 1)


        state = next_state

# Close all the windows
env.close()


### TRAINING ###