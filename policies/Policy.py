'''
Created on Oct 22, 2018

@author: michael
'''
from abc import ABC, abstractmethod
from domains.Task import Task
from agents.Agent import Agent


class Policy(ABC):
    """ Represents an abstract class for an exploration policy used in reinforcement learning.
    """
        
    @abstractmethod
    def clear(self):
        """ Resets all working variables contained in the current implementation.
        """
        pass
    
    @abstractmethod
    def act(self, Q : Agent, task : Task, state):
        """ Selects an action for the specified state in the specified task according 
        to the specified value function. May be either randomized or deterministic.
        
        inputs:
            Q - an Agent object storing the Q-values
            task - a Task object representing the task the agent is learning
            state - the current state of the task for which to select an action
        outputs:
            - the action selected in the specified state
        """
        pass
    
    @abstractmethod
    def distribution(self, Q : Agent, task : Task, state):
        """ For randomized policies, returns a numpy array containing the probability 
        distribution over actions of the current policy for the specified state in the
        specified task according to the specified value function. For deterministic policies,
        this should return a degenerate distribution.
        
        inputs:
            Q - an Agent object storing the Q-values
            task - a Task object representing the task the agent is learning
            state - the current state of the task for which to select an action
        outputs:
            - a one-dimensional numpy array of probabilities of selecting each action
            under the current policy
        """
        pass
    
    @abstractmethod
    def finish_episode(self, episode):
        """ Finishes the current episode.
        
        This method is generally required only in situations where policy parameters
        need to be updated at the end of each episode. For instance, this method is used for:
            - updating the epsilon parameter in epsilon-greedy
            - updating the temperature parameter in Boltzmann exploration
            
        inputs:
            episode - the (zero-based) episode counter
        """
        pass
