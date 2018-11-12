'''
Created on Oct 22, 2018

@author: michael
'''
import random


class ReplayMemory:
    """ A simple and efficient reusable cyclic buffer for randomized experience replay.
    
    inputs:
        memory_capacity - the maximum capacity of the buffer
        batch_size - the size of the random samples generated by the buffer
    """

    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.clear()    
    
    def clear(self):
        """ Removes all data from the buffer.
        """
        self.memory = []
        self.index = 0
        
    def remember(self, state, action, reward, new_state, done):
        """ Stores a new experience in the buffer.
        
        If the current size of the buffer is equal to the maximum capacity of the buffer,
        this method will overwrite the oldest entry in the buffer. Otherwise, this method
        appends the experience to the end of the buffer and its size increases by one.
        
        inputs:
            state - the current state
            action - integer representing the current action
            reward - floating number representing the reward obtained on transition 
            to the next state
            new_state - the next state
            done - boolean whether or not new_state is terminal
        """
        memory = (state, action, reward, new_state, done)
        if len(self.memory) < self.memory_capacity:
            self.memory.append(memory)
        else:
            self.memory[self.index] = memory
        self.index = (self.index + 1) % self.memory_capacity
        
    def sample_batch(self):
        """ If the buffer has enough experiences in memory, this method samples and returns
        a randomly sampled batch of experiences of fixed size from memory.
        
        outputs:
            a list of experiences represented as tuples
        """
        if len(self.memory) < self.batch_size:
            return None
        else:
            return random.sample(self.memory, self.batch_size)
