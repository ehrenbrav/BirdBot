"""Class for doing logistic regression."""
#pylint: disable=C0103

import numpy as np

import theano
import theano.tensor as T

class LogisticClassifier(object):
    """Class for logistic classifier."""

    def __init__(self, input_data, n_in, n_out, init_params=None):

        # Initialize weights with zeros and in the shape n_in x n_out.
        initial_W = None
        if init_params == None:
            initial_W = np.zeros((n_in, n_out), dtype=theano.config.floatX)
        else:
            initial_W = init_params[0]

        # Set up the shared variable.
        self.W = theano.shared(
            value=initial_W,
            name='W',
            borrow=True)

        # Initialize biases as a vector of n_out zeros.
        initial_b = None
        if init_params == None:
            initial_b = np.zeros((n_out), dtype=theano.config.floatX)
        else:
            initial_b = init_params[1]

        # Set up the shared variab.e
        self.b = theano.shared(
            value=initial_b,
            name='b',
            borrow=True)

        # Probability of class y given data x.
        self.p_y_given_x = T.nnet.softmax(
            T.dot(input_data, self.W) + self.b)

        # Get class with highest probability.
        self.y_prediction = T.argmax(
            self.p_y_given_x, axis=1)

        # Parameters of the model.
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Returns the NLL given the input x."""

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Returns the zero-one error, or raises exceptions."""

        if y.ndim != self.y_prediction.ndim:
            raise TypeError(
                "The vector of classes y " +
                "does not have the same shape as " +
                "the vector of predictions y_prediction.",
                ('y', target.type, 'y_prediction', self.y_prediction.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_prediction, y))

        else:
            raise NotImplementedError()

