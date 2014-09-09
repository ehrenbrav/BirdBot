"""
Class for a multi-layer perceptron.
"""
import numpy as np

import theano
import theano.tensor as T

# pylint: disable=C0103,R0903

class HiddenLayer(object):
    """MLP layer used in the models."""

    def __init__(self, data_input, n_in, n_out, init_params=None):
        """
        Initialize all our variables.
        """

        # Initialize the weights.
        W_values = None
        if init_params == None:
            W_values = np.asarray(
                np.random.normal(
                    loc=0.,
                    scale=.01,
                    size=(n_in, n_out)),
                dtype=theano.config.floatX)

        else:
            W_values = init_params[0]

        # Make this into a shared variable.
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize biases and make into a shared variable.
        b_values = None
        if init_params == None:
            b_values = np.ones((n_out,), dtype=theano.config.floatX)
        else:
            b_values = init_params[1]

        # Set up the shared variable.
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # Configure the outputs, handling the linear output special case.
        lin_output = T.dot(data_input, self.W) + self.b
        relu = lambda x: x * (x > 0)
        self.output = relu(lin_output)
        
        # Parameters of the model
        self.params = [self.W, self.b]


