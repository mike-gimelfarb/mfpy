'''
Created on Oct 22, 2018

@author: michael
'''
from abc import abstractmethod
import keras.backend as K
from keras.engine.network import Network
from agents.Agent import Agent


class Neural(Agent):
    """ An abstract class to represent an instance of a deep network architecture
    to represent an instance of the Q-value function. 
    """
    
    @abstractmethod
    def train(self, mini_batch, discount):
        """ Trains the current neural network on a batch of experiences.
        
        inputs:
            mini_batch - an iterable object containing experiences observed
            from a Markov decision process, of the form 
            (state, action, reward, next_state, done)
            discount - the discount factor in [0, 1]
        """
        pass
        
    @staticmethod
    def clear_model(model):
        """ A recursive method to re-initialize all layers in a Keras model.
        
        This method will recursively check all layers in the current model. For each
        layer, if a weight initializer exists, it calls the weight initializer to initialize
        all weights in the layer to their default values. 
        
        inputs:
            model - a Keras model whose weights to re-initialize
        """
        session = K.get_session()
        for layer in model.layers: 
            if isinstance(layer, Network):
                Neural.clear_model(layer)
                continue
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)   
