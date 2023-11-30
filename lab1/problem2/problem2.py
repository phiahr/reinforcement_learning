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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
# from sklearn import preprocessing

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 100        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma
alpha_init = 0.1


# Reward
episode_reward_list = []  # Used to save episodes reward
eta = np.array([[0,1],[1,0],[1,1],[1,2],[2,0],[2,2]])

# eta = np.array([
#     [0,1],
#     [1,0],
#     [1,1],
#     [1,2],
#     [2,1],
#     [2,0]
# ])

A = 3
p = 2
S = 2


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


def phi(state):
    return np.cos(np.pi * np.dot(eta, state))

def Q(state):
    return np.dot(w.T, phi(state))
    

def epsilon_greedy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        a = np.random.randint(0,k)
    else:
        # a = np.argmax(Q[state,:])
        a = np.argmax(Q(state))

    return a


def gradient_descent(r, next_state, next_action, state, action):
    return r + discount_factor * Q(next_state)[next_action] - Q(state)[action]


def gradient_Q(state,action):
    gradient = np.zeros_like(z)
    gradient[:,action] = phi(state)
    return gradient

def reduce_alpha(reward, alpha):
    if reward > -200:
        # return 0.3 * alpha_init
        return 0.3 * alpha0
    else:
        return alpha

# Training process
# w = 0.1*np.random.rand(eta.shape[0], A)
w = np.random.normal(size=(eta.shape[0], A))
z = np.zeros_like(w)
eligibility_param = 0.8
v = np.zeros_like(z)
m = 0.5

epsilon_init = 0.3
# alpha = 0.01/np.linalg.norm(eta, axis=1).reshape(-1,1)
alpha0 = 0.01/np.maximum(np.linalg.norm(eta, axis=1), 1).reshape(-1,1)

for i in tqdm(range(N_episodes)):
    # Reset enviroment data
    done = False
    state, _ = env.reset()
    state = scale_state_variables(state)
    total_episode_reward = 0.
    # epsilon = np.exp(-i / N_episodes)
    epsilon = 0.3 * np.exp(-0.5 * i / N_episodes)
    alpha = alpha0 * np.exp(-0.5 * i / N_episodes)
    # epsilon_factor = 1-(0.1*(i//(0.1*N_episodes)))  # Factor for reducing the exploration rate
    # epsilon = epsilon_init*epsilon_factor
    action = epsilon_greedy(state, epsilon)
    while not done:
        # Take a random action
        # env.action_space.n tells you the number of actions
        # available
        # action = np.random.randint(0, k)
            
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _, _ = env.step(action)
        next_action = epsilon_greedy(state, epsilon)
        next_state = scale_state_variables(next_state)
        delta = gradient_descent(reward, next_state, next_action, state, action)
        z += discount_factor * eligibility_param * z 
        if action == next_action:
            z += gradient_Q(state, action)
        z = np.clip(z,-5,5)

        v = m*v + alpha*delta*z

        w += m*v + alpha*delta*z

        # Update episode reward
        total_episode_reward += reward
            
        # Update state for next iteration
        state = next_state
        action = next_action

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    # alpha = reduce_alpha(reward, alpha)

    # Close environment
    env.close()
    

# Plot Rewards
plt.plot([i for i in range(10, N_episodes+1)], episode_reward_list[9:], label='Episode reward')
plt.plot([i for i in range(10, N_episodes+1)], running_average(episode_reward_list[9:], 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()