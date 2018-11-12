'''
Created on Sep 19, 2018

@author: michael
'''
from abc import ABC, abstractmethod


class Task(ABC):
    """ An abstract class that represents a general Markov decision process 
    (MDP) environment.
    """

    @abstractmethod
    def initial_state(self):
        """ Returns the initial state of the environment. May be either
        deterministic or randomly generated. 
        
        outputs:
            the initial state
        """
        pass

    @abstractmethod
    def valid_actions(self):
        """ Returns an integer representing the total number of valid actions
        available to an agent when interacting with this environment.
        
        outputs:
            an integer counting the total number of valid actions
        """
        pass

    @abstractmethod
    def transition(self, state, action):
        """ Performs a transition in the environment to a new state from a specified
        current state and when a specified action is taken.
        
        inputs:
            state - the current state of the environment
            action - the current action taken
        outputs:
            a triple (new_state, reward, done), where:
                new_state - the next state the system has transitioned to
                reward - floating number representing the reward obtained upon transition
                done - boolean whether or not new_state is terminal
        """
        pass
