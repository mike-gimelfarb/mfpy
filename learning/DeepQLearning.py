'''
Created on Oct 22, 2018

@author: michael
'''
import numpy as np
from domains.Task import Task
from learning.TDLearning import TDLearning
from policies.Policy import Policy
from agents.ReplayMemory import ReplayMemory
from agents.Neural import Neural


class DeepQLearning(TDLearning):
    """ Represents the algorithm for Q-learning using function approximation. 
    
    This class currently is a representation for both DQN and double DQN (DDQN). 
    In other words, if the agent is an instance of DeepQ, this will perform DQN learning, 
    and if the agent is an instance of DoubleDeepQ, will perform double DQN learning. 
    
    In future implementations, it is hoped that this class definition will be 
    further generalized to handle prioritized experience replay (among other variations), 
    as well as other kinds of agents, such as A3C, policy gradients, and others.
    
    inputs:
        discount - the discount factor in [0, 1]
        episode_length - for episodic learning, the length of each episode
        state_encoding - a lambda expression to encode the state into a vector form
        which can serve as input to a neural network
        memory - an instance of ReplayMemory that serves as a replay buffer to perform
        offline training of the DQN on batches of experiences
    """
    
    def __init__(self, discount, episode_length, state_encoding, memory : ReplayMemory):
        super().__init__(discount, episode_length)
        self.phi = state_encoding
        self.memory = memory
        self.clear()
    
    def clear(self):
        self.memory.clear()
        
    def run_episode(self, Q : Neural, task : Task, policy : Policy):
        
        # to compute backup
        rewards = np.zeros(self.episode_length, dtype=float)
        
        # initialize state
        state = task.initial_state()
        phi_state = self.phi(state)
            
        # repeat for each step of episode
        for t in range(self.episode_length):
                
            # choose action from state using policy derived from Q
            action = policy.act(Q, task, phi_state) 
                
            # take action and observe reward and new state
            new_state, reward, done = task.transition(state, action) 
            phi_new_state = self.phi(new_state)
            rewards[t] = reward  
               
            # store the transition in memory
            self.memory.remember(phi_state, action, reward, phi_new_state, done)
            
            # sample a mini-batch of transitions from memory
            # compute the targets y_j and train the network
            mini_batch = self.memory.sample_batch()
            if mini_batch is not None:            
                Q.train(mini_batch, self.gamma)
            
            # update state
            state, phi_state = new_state, phi_new_state
            
            # until state is terminal
            if done:
                break

        return t, rewards[0:t]
    