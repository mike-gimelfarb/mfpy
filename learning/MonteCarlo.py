'''
Created on Oct 22, 2018

@author: michael
'''
from learning.TDLearning import TDLearning
import numpy as np
from domains.Task import Task
from policies.Policy import Policy
from agents.Tabular import Tabular


class MonteCarlo(TDLearning):
    """ Represents the tabular first-visit offline Monte Carlo algorithm for control.
    
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
        
        # simulate a trajectory
        t, episode = self.sample_episode(Q, task, policy, rewards)
        
        # repeat for each step of episode
        G = 0.0
        for state, action, reward in episode[::-1]:
               
            # update cumulative reward
            G = reward + self.gamma * G
             
            # update Q
            delta = G - Q.values(state)[action]
            Q.update(state, action, delta)
                
        return t, rewards[0:t]
    
    def sample_episode(self, Q : Tabular, task : Task, policy : Policy, rewards):
        
        # to store the episode
        episode = [None] * (self.episode_length)
        
        # initialize state
        state = task.initial_state()
        
        # repeat for each step of episode
        for t in range(self.episode_length):
            
            # choose action from state using policy derived from Q
            action = policy.act(Q, task, state)
            
            # take action and observe reward and new state
            new_state, reward, done = task.transition(state, action)
            rewards[t] = reward
            episode[t] = (state, action, reward)
            
            # update state
            state = new_state
            
            # until state is terminal
            if done:
                break
        
        return t, episode[0:t]
