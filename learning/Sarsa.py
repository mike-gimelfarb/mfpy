'''
Created on Oct 22, 2018

@author: michael
'''
from learning.TDLearning import TDLearning
import numpy as np
from domains.Task import Task
from policies.Policy import Policy
from agents.Tabular import Tabular


class Sarsa(TDLearning):
    """ Represents the tabular online Sarsa algorithm.
    
    inputs:
        discount - the discount factor in [0, 1]
        episode_length - for episodic learning, the length of each episode
        
    References
    ========
        - Sutton, Richard S., and Andrew G. Barto. 
        Reinforcement learning: An introduction. MIT press, 2018.
    """
    
    def __init__(self, discount, episode_length):
        super().__init__(discount, episode_length)
    
    def clear(self):
        pass
    
    def run_episode(self, Q : Tabular, task : Task, policy : Policy):
        
        # to compute backup
        rewards = np.zeros(self.episode_length, dtype=float)
        
        # initialize state
        state = task.initial_state()
        
        # choose action from state using policy derived from Q
        action = policy.act(Q, task, state) 
        
        # repeat for each step of episode
        for t in range(self.episode_length):
                
            # take action and observe reward and new state
            new_state, reward, done = task.transition(state, action) 
            rewards[t] = reward  
                
            # choose new action from new state using policy derived from Q
            new_action = policy.act(Q, task, new_state) 
            
            # update Q
            delta = reward + self.gamma * Q.values(new_state)[new_action] - Q.values(state)[action]
            Q.update(state, action, delta)
                
            # update state and action
            state, action = new_state, new_action
            
            # until state is terminal
            if done:
                break

        return t, rewards[0:t]
