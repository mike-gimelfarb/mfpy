'''
Created on Oct 22, 2018

@author: michael
'''
from abc import ABC, abstractmethod
import numpy as np
from domains.Task import Task
from agents.Agent import Agent
from policies.Policy import Policy


class TDLearning(ABC):
    """ An abstract class that represents all temporal difference learning algorithms.
    
    inputs:
        discount - the discount factor in [0, 1]
        episode_length - for episodic learning, the length of each episode
    """
    
    def __init__(self, discount, episode_length):
        self.gamma = discount
        self.episode_length = episode_length
    
    @abstractmethod
    def clear(self):
        """ Resets all working variables contained in the current implementation.
        """
        pass
        
    @abstractmethod
    def run_episode(self, Q : Agent, task : Task, policy : Policy):
        """ Trains the specified agent on the specified task using the specified
        exploration policy using the current implementation. One episode is generated
        and used for training.
        
        inputs:
            Q - an Agent object storing the Q-values
            task - a Task object representing the task the agent is learning
            policy - a Policy object representing the exploration policy used to 
            balance exploration and exploitation
        outputs:
            - the length of the episode until the terminal state is reached
            or the episode reaches its maximum length
            - a one-dimensional numpy array containing the sum of the discounted 
            rewards from the environment
        """
        pass    
    
    def train(self, Q : Agent, task : Task, policy : Policy, episodes):
        """ Trains the specified agent on the specified task using the specified
        exploration policy using the current implementation. A specified number of episodes
        is generated for training.
        
        inputs:
            Q - an Agent object storing the Q-values
            task - a Task object representing the task the agent is learning
            policy - a Policy object representing the exploration policy used to 
            balance exploration and exploitation
            episodes - the number of episodes of training to perform
        outputs:
            - a one-dimensional numpy array containing the lengths of each episode - this
            can be used to check the learning progress of the agent
            - a one-dimensional numpy array containing the sum of the discounted 
            rewards from the environment obtained on each episode - this can be used to check
            the learning progress of the agent 
        """
        
        # initialization
        self.clear()
        Q.clear()
        policy.clear()
        
        # for storing history of trial
        rewards_history = np.zeros(episodes, dtype=float)
        steps_history = np.zeros(episodes, dtype=int)
        
        # run episodes
        for e in range(episodes):
            
            # run an episode of training
            steps, rewards = self.run_episode(Q, task, policy)
            
            # compute the value of the backup and update the history
            R = 0.0
            for reward in rewards[::-1]:
                R = reward + self.gamma * R
            rewards_history[e] = R
            steps_history[e] = steps
            
            # finish episode
            policy.finish_episode(e)
            Q.finish_episode(e)
        
        return steps_history, rewards_history
    
    def train_many(self, Q : Agent, task : Task, policy : Policy, episodes, trials):
        """ Trains the specified agent on the specified task using the specified
        exploration policy using the current implementation. A specified number of episodes
        is generated for training. A specified number of independent trials of training are
        performed - all variables are reset at the end of each trial. 
        
        inputs:
            Q - an Agent object storing the Q-values
            task - a Task object representing the task the agent is learning
            policy - a Policy object representing the exploration policy used to 
            balance exploration and exploitation
            episodes - the number of episodes of training to perform
            trials - the number of independent trials of training to perform
        outputs:
            - a one-dimensional numpy array containing the average length of each episode 
            over all trials - this can be used to check the learning progress of the agent
            - a one-dimensional numpy array containing the average of the sum of the discounted 
            rewards from the environment obtained on each episode over all trials - this can 
            be used to check the learning progress of the agent 
        """
        average_rewards = np.zeros(episodes, dtype=float)
        average_steps = np.zeros(episodes, dtype=float)
        for _ in range(trials):
            steps, rewards = self.train(Q, task, policy, episodes)
            average_rewards += rewards / trials
            average_steps += steps / trials
        return average_steps, average_rewards
