'''
Created on Oct 22, 2018

@author: michael
'''
import numpy as np
import random
from policies.Policy import Policy
from domains.Task import Task
from agents.Agent import Agent


class EpsilonGreedy(Policy):
    """ Represents the epsilon-greedy exploration policy. Allows both constant or changing
    epsilon parameter over time.
    
    inputs:
        epsilon - the probability of selecting a random action - may be either a lambda
        expression of one parameter (the episode counter), or a floating point number
    """
    
    def __init__(self, epsilon):
        if isinstance(epsilon, float):
            self.epsilon_lambda = lambda e: epsilon
        else:
            self.epsilon_lambda = epsilon
        self.clear()
    
    def clear(self):
        self.epsilon = self.epsilon_lambda(0)
        
    def distribution(self, Q : Agent, task : Task, state):
        num_actions = task.valid_actions()
        values = np.full(num_actions, self.epsilon / num_actions)
        values[Q.max_action(state)] += 1.0 - self.epsilon
        return values
        
    def act(self, Q : Agent, task : Task, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(task.valid_actions())
        else:
            return Q.max_action(state)
        
    def finish_episode(self, episode):
        self.epsilon = self.epsilon_lambda(episode)
