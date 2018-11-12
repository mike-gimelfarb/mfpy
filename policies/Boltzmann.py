'''
Created on Oct 22, 2018

@author: michael
'''
import numpy as np
from policies.Policy import Policy
from domains.Task import Task
from agents.Agent import Agent


class Boltzmann(Policy):
    """ Represents the Boltzmann exploration policy. Allows both constant or changing
    temperature over time.
    
    inputs:
        temperature - the temperature parameter - may be either a lambda expression
        of one integer parameter (the episode counter), or a floating point number
    """
    
    def __init__(self, temperature):
        if isinstance(temperature, float):
            self.temperature = lambda e: temperature
        else:
            self.temperature = temperature
        self.clear()
    
    def clear(self):
        self.temp = self.temperature(0)
        
    def distribution(self, Q : Agent, task : Task, state):
        values = Q.values(state)
        values = np.exp(values / self.temp)
        values /= np.sum(values)
        return values
        
    def act(self, Q : Agent, task : Task, state):
        values = self.distribution(Q, task, state)
        return np.random.choice(task.valid_actions(), 1, p=values)
        
    def finish_episode(self, episode):
        self.temp = self.temperature(episode)
        