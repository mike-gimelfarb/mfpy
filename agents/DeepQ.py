'''
Created on Oct 22, 2018

@author: michael
'''
import numpy as np
from agents.Neural import Neural


class DeepQ(Neural):
    """ A simple class that represents a deep Q network (DQN), 
    implemented using a Keras model.
    
    inputs:
        network - a Keras model for the neural network
        train_epochs - the number of epochs of training that is done per batch
    
    References
    ========
        - Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." 
        arXiv preprint arXiv:1312.5602 (2013).
        - Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 
        Nature 518.7540 (2015): 529.
    """

    def __init__(self, network, train_epochs):
        self.network = network
        self.train_epochs = train_epochs
        self.clear()

    def clear(self):
        Neural.clear_model(self.network)
        self.state_placeholder = None
        self.next_state_placeholder = None
        
    def values(self, state):
        return self.network.predict(state)[0]
        
    def finish_episode(self, episode):
        pass
    
    def train(self, mini_batch, discount):
        
        # get the states at which to predict Q-values
        if self.state_placeholder is None:
            state_dim = mini_batch[0][0].size
            self.state_placeholder = np.empty((len(mini_batch), state_dim))
            self.next_state_placeholder = np.empty((len(mini_batch), state_dim))
        for i, (state, _, _, new_state, _) in enumerate(mini_batch):
            self.state_placeholder[i] = state
            self.next_state_placeholder[i] = new_state
        
        # get the Q-values
        values = self.network.predict(self.state_placeholder)
        next_values = self.network.predict(self.next_state_placeholder)
        
        # compute the new Q-value targets using Q-learning
        for i, (_, action, reward, _, done) in enumerate(mini_batch):
            if done:
                values[i][action] = reward
            else:
                values[i][action] = reward + discount * np.amax(next_values[i])
        
        # fit the network on the data
        self.network.fit(self.state_placeholder, values,
                         batch_size=len(mini_batch),
                         epochs=self.train_epochs,
                         verbose=0)
