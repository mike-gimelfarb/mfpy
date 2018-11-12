'''
Created on Sep 19, 2018

@author: michael
'''
from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """ An abstract class to represent an instance of the Q-value function.
    """
    
    @abstractmethod
    def clear(self):
        """ Re-initialize the Q-value function to its default values.
        """
        pass
    
    @abstractmethod
    def values(self, state):
        """ Returns a numpy array containing the Q-values for all actions
        for the specified state.
        
        inputs:
            state - the state for which to compute Q-values
        outputs:
            a one-dimensional numpy array of Q-values
        """
        pass
    
    @abstractmethod    
    def finish_episode(self, episode):
        """ Finishes the current episode.
        
        This method is generally required only in situations where learning parameters
        need to be updated at the end of each episode. For instance, this method is used for:
            tabular methods - the learning rate is updated at the end of each episode
            double deep Q - the weights of the action network are updated to those of the 
            target network at the end of each episode
        
        inputs:
            episode - the (zero-based) episode counter
        """
        pass

    def max_action(self, state):
        """ Returns the index of the action that attains the maximum of the Q-values
        for the specified state.
        
        inputs:
            state - the state for which to compute Q-values
        outputs:
            an integer index representing the action which maximizes the Q-values for the
            given state
        """        
        return np.argmax(self.values(state))
    
    def max_value(self, state):
        """ Returns the maximum of the Q-values over all actions for the specified state.
        
        inputs:
            state - the state for which to compute Q-values
        outputs:
            a floating point number representing the maximum of the Q-values for the 
            given state
        """
        return np.amax(self.values(state))
