'''
Created on Oct 14, 2018

@author: michael
'''
from collections import defaultdict
import numpy as np
import math
from agents.Agent import Agent


class Tabular(Agent):
    """ A hash-table implementation of a Q-value table. 
    
    States are stored as keys (and therefore must be hashable). Action values are stored
    as one-dimensional numpy arrays. This guarantees constant time reading and writing.
    
    inputs:
        valid_actions - the number of possible actions
        learning_rate - the learning rate used to update the table
        clip_min - minimum value that can be used to update the table 
        (defaults to negative infinity)
        clip_max - maximum value that can be used to update the table
        (defaults to positive infinity)
        randomizer - numpy method handle to initialize action values
        (defaults to np.zeros)
    """

    def __init__(self, valid_actions, learning_rate,
                 clip_min=-math.inf, clip_max=math.inf, randomizer=np.zeros):
        self.valid_actions = valid_actions
        self.randomizer = randomizer
        self.clip_min = clip_min
        self.clip_max = clip_max
        if isinstance(learning_rate, float):
            self.learning_rate = lambda e: learning_rate
        else:
            self.learning_rate = learning_rate
        self.clear()
        
    def clear(self):
        self.alpha = self.learning_rate(0)
        self.Q = defaultdict(lambda: self.randomizer(self.valid_actions))
    
    def values(self, state):
        return self.Q[state]
     
    def finish_episode(self, episode):
        self.alpha = self.learning_rate(episode)
    
    def update(self, state, action, error): 
        """ Updates the Q-value for a specified state-action pair and Bellman error.
        
        The formula is Q[state, action] += learning_rate * error.
        
        inputs:
            state - the current state
            action - the current selected action
            error - the Bellman error
        """
        change = self.alpha * error
        change = max(min(change, self.clip_max), self.clip_min)
        self.Q[state][action] += change
    
    def update_all(self, state, errors):
        """ Updates all Q-values for a specified state and array of Bellman errors.
        
        The formula is Q[state] += learning_rate * errors.
        
        inputs:
            state - the current state 
            errors - a one-dimensional numpy array of Bellman errors for each action; 
            must be of the same length as valid_actions
        """
        change = self.alpha * errors
        change = np.clip(change, self.clip_min, self.clip_max, change)
        self.Q[state] += change
