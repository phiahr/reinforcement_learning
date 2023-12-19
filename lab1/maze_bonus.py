# Authors: 
# * Oscar Eriksson, 0011301991, oscer@kth.se
# * Philip Ahrendt, 960605R119, pcah@kth.se

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
from tqdm import tqdm


# Implemented methods
methods = ['DynProg', 'ValIter', 'qLearn'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
RED          = '#FF0000';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    MINOTAUR_STAY = False
    MOVE_TO_PLAYER = False


    def __init__(self, maze, minotaur_stay=False, move_to_player=False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.MINOTAUR_STAY            = minotaur_stay;
        self.MOVE_TO_PLAYER           = move_to_player;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.minotaur_actions         = self.__minotaur_actions()
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();
    
        

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __minotaur_actions(self):
        actions = dict();
        
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        if self.MINOTAUR_STAY:
            actions[self.STAY]   = (0, 0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        for key in range(2):
                            if self.maze[i,j] != 1 :
                                states[s] = (i,j,k,l,key);
                                map[(i,j,k,l,key)] = s;
                                s += 1;
                        # if self.maze[i,j] != 1 :
                        #     states[s] = (i,j,k,l);
                        #     map[(i,j,k,l)] = s;
                        #     s += 1;
        return states, map
     
    def __minotaur_move(self, state):
        minotaur_positions = []
        for action in self.minotaur_actions:
            row = self.states[state][2] + self.minotaur_actions[action][0];
            col = self.states[state][3] + self.minotaur_actions[action][1];
            hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1])
            if not hitting_maze_walls:
                minotaur_positions.append((row,col))
        return minotaur_positions

    def move(self, state, action):
        return self.__move(state, action)
    
    def __move(self, state, action):
        # Caught by the minotaur
        if self.states[state][0] == self.states[state][2] and self.states[state][1] == self.states[state][3]:
            return [state]

        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        key = self.states[state][4]
        
        minotaur_positions = self.__minotaur_move(state)



        # Is the future position an impossible one ?
        hitting_maze_walls =  (row <= -1) or (row >= self.maze.shape[0]) or (col <= -1) or \
               (col >= self.maze.shape[1]) or (self.maze[row,col] == 1)

        
        next_states = []
        if hitting_maze_walls:
            # return state;
            for minotaur_position in minotaur_positions:
                next_states.append(self.map[(self.states[state][0], self.states[state][1], minotaur_position[0], minotaur_position[1], key)])
        else:
            if self.maze[row, col] == 3:
                key = 1
            for minotaur_position in minotaur_positions:
                next_states.append(self.map[(row, col, minotaur_position[0], minotaur_position[1], key)])
        return next_states

    def __transitions(self):
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.__move(s,a)
                if self.MOVE_TO_PLAYER:
                    points = [self.states[s] for s in next_states]
                    points = np.array(points)
                    index_of_closest_point = np.argmin(np.linalg.norm(points[:,2:4] - points[:,:2], axis=1))
                    for index, next_s in enumerate(next_states):
                        # the minotaur moves with probability 35% towards you and with probability 65% uniformly at random in all directions
                        if len(next_states)>1:
                            if index == index_of_closest_point:
                                transition_probabilities[next_s, s, a] = 0.35
                            else:
                                transition_probabilities[next_s, s, a] = (1/(len(next_states)-1))*0.65
                        else:
                            transition_probabilities[next_s, s, a] = 1                    
                else:
                    for next_s in next_states:
                        transition_probabilities[next_s, s, a] = 1/len(next_states)
        return transition_probabilities

    def __rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.__move(s,a)
                # Compute the average reward for action (s,a).
                reward = 0
                for next_s in next_states:
                    if self.states[s][:2] == self.states[next_s][:2] and a != self.STAY:
                        reward += self.IMPOSSIBLE_REWARD * self.transition_probabilities[next_s, s, a]
                    if self.states[s][:2] == self.states[s][2:4]:
                        reward += self.IMPOSSIBLE_REWARD * self.transition_probabilities[next_s, s, a]
                    if self.maze[self.states[next_s][0:2]] == 3 and self.states[next_s][4] == 0:
                        reward += self.GOAL_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for reaching the exit with key
                    elif self.states[s][0:2] == self.states[next_s][0:2] and self.maze[self.states[next_s][0:2]] == 2 and self.states[s][4] == 1:
                        reward += self.GOAL_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        reward += self.STEP_REWARD * self.transition_probabilities[next_s, s, a]
                rewards[s,a] = reward
        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':

            t_death = 50
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                
                # Add the position in the maze corresponding to the next state
                # to the path
                next_s = random.choice(next_s)
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path

            # with probability 1/30 the player will die from poison
            dead_from_poison = False
            # if random.randint(1,30) == 1:
            #     dead_from_poison = True


            # if not dead_from_poison:
            next_s = random.choice(next_s)
            # else:
            #     next_s = start
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                if random.randint(1,30) == 1:
                    dead_from_poison = True
                    break

                if not dead_from_poison:
                    next_s = self.__move(s,policy[s]);
                else:
                    next_s = s;
                # Add the position in the maze corresponding to the next state
                # to the path
                next_s = random.choice(next_s)
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        if method == 'qLearn':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_states = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path

            # walk 0.35 towards player
            if self.MOVE_TO_PLAYER:
                points = [self.states[s] for s in next_states]
                points = np.array(points)
                index_of_closest_point = np.argmin(np.linalg.norm(points[:,2:4] - points[:,:2], axis=1))

                # print(test)
                det_state = next_states[index_of_closest_point]
                next_states.pop(index_of_closest_point)
                # next_s = random.choices[test]
                if random.random() <= 0.35:
                    next_s = det_state
                else:
                    next_s = random.choice(next_states)
            else:
                next_s = random.choice(next_states)
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            t_death = 50
            while s != next_s and t < 200:
                # Update state
                s = next_s;
                next_states = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                if self.MOVE_TO_PLAYER:
                    points = [self.states[s] for s in next_states]
                    points = np.array(points)
                    index_of_closest_point = np.argmin(np.linalg.norm(points[:,2:4] - points[:,:2], axis=1))

                    # print(test)
                    det_state = next_states[index_of_closest_point]
                    if len(next_states) > 1:
                        next_states.pop(index_of_closest_point)
                    # next_s = random.choices[test]
                    if random.random() <= 0.35:
                        next_s = det_state
                    else:
                        next_s = random.choice(next_states)
                else:
                    next_s = random.choice(next_states)
                
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        
        return path, t_death


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

    def choose_state(self, states, next_states):
        points = [states[s] for s in next_states]
        points = np.array(points)
        index_of_closest_point = np.argmin(np.linalg.norm(points[:,2:4] - points[:,:2], axis=1))

        # print(test)
        det_state = next_states[index_of_closest_point]
        if len(next_states) > 1:
            next_states.pop(index_of_closest_point)
        # next_s = random.choices[test]
        if random.random() <= 0.35:
            next_s = det_state
        else:
            next_s = random.choice(next_states)
        return next_s

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;



def q_learning(env, start, gamma, epsilon, alpha, num_episodes, Q_Hot = None):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :input float alpha        : learning rate
        :input int num_episodes   : number of episodes
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    if Q_Hot is None:
        Q = np.zeros((n_states, n_actions));
    else:
        Q = Q_Hot
    Vs = []
    n_visits = np.zeros((n_states, n_actions))
    for n in tqdm(range(num_episodes)):
        t = 0
        s = env.map[start]

        Vs.append(np.max(Q[s, :]))

        # epsilon greedy policy
        if random.uniform(0,1) < epsilon:
            a = random.randint(0,4)
        else:
            a = np.argmax(Q[s,:])
        while t < 200:
            # epsilon greedy policy
            if random.uniform(0,1) < epsilon:
                a = random.randint(0,4)
            else:
                a = np.argmax(Q[s,:])
        
            next_states = env.move(s,a)
            # next_states = env.__move()
            # next_s = random.choice(next_states)
            next_s = env.choose_state(env.states, next_states)
            n_visits[s,a] += 1
            step_size = 1/(n_visits[s,a]**alpha)

            Q[s, a] = Q[s, a] + step_size * (r[s, a] + gamma*np.max(Q[next_s,:]) - Q[s, a])
           
            if env.maze[env.states[s][0:2]] == 2 and env.states[s][4] == 1:
                break
            elif env.states[s][0:2] == env.states[s][2:4]:
                break
            t += 1
            s = next_s
            V = np.max(Q, 1)
    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy, Vs, Q;

def sarsa(env, start, gamma, epsilon, alpha, num_episodes, decreasing_epsilon=False, delta=0.55):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :input float alpha        : learning rate
        :input int num_episodes   : number of episodes
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    # Iteration counter

    Vs = []
    n_visits = np.zeros((n_states, n_actions))
    for n in tqdm(range(num_episodes)):
        t = 0
        s = env.map[start]
        
        if decreasing_epsilon and n > 0:
            epsilon = 1/n**delta

        Vs.append(np.max(Q[s, :]))

        # epsilon greedy policy
        if random.uniform(0,1) < epsilon:
            a = random.randint(0,4)
        else:
            a = np.argmax(Q[s,:])
        prev_s = -1
        while t < 200:
            # epsilon greedy policy
            if random.uniform(0,1) < epsilon:
                a = random.randint(0,4)
            else:
                a = np.argmax(Q[s,:])
        
            next_states = env.move(s,a)
            # next_states = env.__move()
            next_s = random.choice(next_states)

            if random.uniform(0,1) < epsilon:
                next_a = random.randint(0,4)
            else:
                next_a = np.argmax(Q[next_s,:])

            n_visits[s,a] += 1
            step_size = 1/(n_visits[s,a]**alpha)

            Q[s, a] = Q[s, a] + step_size * (r[s, a] + gamma*Q[next_s,next_a] - Q[s, a])
        
            if env.maze[env.states[s][0:2]] == 2 and env.states[s][4] == 1:
                break
            elif env.states[s][0:2] == env.states[s][2:4]:
                break
            t += 1
            s = next_s
            a = next_a
            V = np.max(Q,1)
    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy, Vs, Q;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: LIGHT_PURPLE};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2:4])].set_facecolor(RED)
        grid.get_celld()[(path[i][2:4])].get_text().set_text('Minotaur')
        
        if i > 0:
            if path[i][:2] == (6,5) and path[i][4] == 1:
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][:2])].get_text().set_text('Player is out')
                grid.get_celld()[(path[i][2:4])].set_facecolor(col_map[maze[path[i][2:4]]])
                grid.get_celld()[(path[i][2:4])].get_text().set_text('')
            elif path[i][:2] == path[i-1][:2]:
                # grid.get_celld()[(path[i][2:4])].set_facecolor(col_map[maze[path[i][2:4]]])
                grid.get_celld()[(path[i][:2])].get_text().set_text('Wait')
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[maze[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')
            else:
                grid.get_celld()[(path[i-1][:2])].set_facecolor(col_map[maze[path[i-1][:2]]])
                grid.get_celld()[(path[i-1][:2])].get_text().set_text('')
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[maze[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')
                # if minotaur_path[i] == minotaur_path[i-1]:
                #     grid.get_celld()[(minotaur_path[i])].set_facecolor(RED)
                #     grid.get_celld()[(minotaur_path[i])].get_text().set_text('Minotaur')
                # else:
                #     grid.get_celld()[(minotaur_path[i-1])].set_facecolor(col_map[maze[minotaur_path[i-1]]])
                #     grid.get_celld()[(minotaur_path[i-1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
