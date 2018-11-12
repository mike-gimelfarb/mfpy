'''
Created on Oct 22, 2018

@author: michael
'''
from collections import defaultdict
import numpy as np
from agents.Tabular import Tabular
from domains.Task import Task
from learning.TDLearning import TDLearning
from policies.Policy import Policy


class OnlineSarsaLambda(TDLearning):
    """ Represents the tabular online Sarsa-Lambda algorithm implemented 
    using eligibility traces.
    
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
        
        # initialize the e(s, a) matrix
        # note: there is an error in Sutton and Barto since e is reset each episode
        e = defaultdict(lambda: np.zeros(task.valid_actions(), dtype=float))
        
        # initialize state and action
        state = task.initial_state()
        action = policy.act(Q, task, state)
        
        # repeat for each step of episode
        for t in range(self.episode_length):
                
            # take action and observe reward and new state
            new_state, reward, done = task.transition(state, action) 
            rewards[t] = reward  
                   
            # choose action from state using policy derived from Q
            new_action = policy.act(Q, task, new_state) 
             
            # update e
            e[state][action] += 1.0 
            
            # update trace
            delta = reward + self.gamma * Q.values(new_state)[new_action] - Q.values(state)[action]
            for s in e.keys():
                errors = e[s] * delta
                Q.update_all(s, errors)
                e[s] *= self.gamma * self.decay

            # update state and action
            state, action = new_state, new_action
            
            # until state is terminal
            if done:
                break

        return t, rewards[0:t]
