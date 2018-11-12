'''
Created on Nov 1, 2018

@author: michael
'''
import numpy as np
from agents.Tabular import Tabular
from domains.Task import Task
from learning.TDLearning import TDLearning
from policies.Policy import Policy


class OfflineSarsaLambda(TDLearning):
    """ Represents the tabular offline Sarsa-Lambda algorithm.
    
    inputs:
        discount - the discount factor in [0, 1]
        episode_length - for episodic learning, the length of each episode
        decay - the lambda parameter in [0, 1] (defaults to 0.9)
        
    References
    ========
        - Sutton, Richard S., and Andrew G. Barto. 
        Reinforcement learning: An introduction. MIT press, 2018.
    """
    
    def __init__(self, discount, episode_length, decay=0.9):
        super().__init__(discount, episode_length)
        self.decay = decay
        
    def clear(self):
        pass
    
    def run_episode(self, Q : Tabular, task : Task, policy : Policy):
        
        # to compute backups
        rewards = np.zeros(self.episode_length, dtype=float)
        states = [None] * self.episode_length
        actions = [None] * self.episode_length
        
        # initialize state
        state = task.initial_state()
        
        # repeat for each step of episode
        for t in range(self.episode_length):
            
            # choose action from state using policy derived from Q
            action = policy.act(Q, task, state)
            states[t], actions[t] = state, action
            
            # take action and observe reward and new state
            new_state, reward, done = task.transition(state, action) 
            rewards[t] = reward  
            
            # update state and action
            state = new_state
            
            # until state is terminal
            if done:
                break
        
        # initialize lambda-average of returns
        lambda_return = reward
        
        # repeat for each step of episode in reverse order 
        T = t
        for t in range(T, -1, -1):
            
            # compute lambda-average of returns
            if t < T:
                lambda_return = rewards[t] + self.gamma * (
                    (1.0 - self.decay) * Q.values(states[t + 1])[actions[t + 1]] + 
                    self.decay * lambda_return)
            
            # update Q
            delta = lambda_return - Q.values(states[t])[actions[t]]
            Q.update(states[t], actions[t], delta)
            
        return T, rewards[0:T]
