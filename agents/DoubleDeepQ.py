'''
Created on Oct 22, 2018

@author: michael
'''
from keras.models import clone_model
import numpy as np
from agents.Neural import Neural


class DoubleDeepQ(Neural):
    """ A simple class that represents a double Deep Q network (DDQN), 
    implemented using Keras models.
    
    inputs:
        network - a Keras model that will be used for both action and target networks
        train_epochs - the number of epochs of training that is done per batch
    
    References
    ========
        - Van Hasselt, Hado, Arthur Guez, and David Silver. 
        "Deep Reinforcement Learning with Double Q-Learning." AAAI. Vol. 2. 2016.
    """

    def __init__(self, network, train_epochs):
        self.target_network = network
        self.action_network = clone_model(network)
        self.action_network.set_weights(network.get_weights())
        self.train_epochs = train_epochs
        self.clear()

    def clear(self):
        Neural.clear_model(self.action_network)
        Neural.clear_model(self.target_network)
        self.state_placeholder = None
        self.next_state_placeholder = None
        
    def values(self, state):
        return self.target_network.predict(state)[0]
        
    def finish_episode(self, episode):
        self.action_network.set_weights(self.target_network.get_weights())
            
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
        values = self.target_network.predict(self.state_placeholder)
        next_values = self.target_network.predict(self.next_state_placeholder)
        next_action_values = self.action_network.predict(self.next_state_placeholder)
            
        # compute the new Q-value targets using Q-learning
        for i in range(len(mini_batch)):
            state, action, reward, _, done = mini_batch[i]
            if done:
                values[i][action] = reward
            else:
                a = np.argmax(next_action_values[i])
                values[i][action] = reward + discount * next_values[i][a]
        
        # fit the network on the data
        self.target_network.fit(self.state_placeholder, values,
                                batch_size=len(mini_batch),
                                epochs=self.train_epochs,
                                verbose=0)
