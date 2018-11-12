'''
Created on Oct 27, 2018

@author: michael
'''
from collections import defaultdict
import numpy as np
from policies.Policy import Policy
from domains.Task import Task
from agents.Agent import Agent


class Pursuit(Policy):
    """ Represents the pursuit exploration policy. Allows both constant or changing
    learning rate over time.
    
    inputs:
        learning_rate - the learning rate parameter - may be either a lambda expression
        of one integer parameter (the episode counter), or a floating point number
    """
    
    def __init__(self, learning_rate):
        if isinstance(learning_rate, float):
            self.beta_lambda = lambda e: learning_rate
        else:
            self.beta_lambda = learning_rate
        self.clear()
        
    def clear(self):
        self.valid_actions = 0
        self.beta = self.beta_lambda(0)
        self.preferences = defaultdict(
            lambda: np.ones(self.valid_actions) / self.valid_actions)
    
    def distribution(self, Q : Agent, task : Task, state):
        return self.preferences[state]
    
    def act(self, Q : Agent, task : Task, state):
        
        # set the number of actions of the current task, if not set
        if self.valid_actions == 0:
            self.valid_actions = task.valid_actions()
            
        # get the distribution over actions for the current state
        pref = self.preferences[state]
        
        # sample an action from the preference distribution
        action = np.random.choice(self.valid_actions, 1, p=pref)
        
        # get the greedy action according to Q
        greedy = Q.max_action(state)
        
        # update the preference distribution
        pref *= (1.0 - self.beta)
        pref[greedy] /= (1.0 - self.beta)
        pref[greedy] += self.beta * (1.0 - pref[greedy])
        
        return action
        
    def finish_episode(self, episode):
        self.beta = self.beta_lambda(episode)
