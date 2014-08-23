"""Class for doing logistic regression."""
#pylint: disable=C0103

import numpy as np

import theano
import theano.tensor as T

class LogisticClassifier(object):
    """Class for logistic classifier."""

    def __init__(self, input_data, n_in, n_out):

        # Initialize weights with zeros and in the shape n_in x n_out.
        self.weights = theano.shared(
            value=np.zeros((n_in, n_out),
                           dtype=T.config.floatX),
                           name='W',
            borrow=True)

        # Initialize biases as a vector of n_out zeros.
        self.biases = theano.shared(
            value=np.zeros((n_out,),
                           dtype=theano.config.floatX),
                           name='b',
            borrow=True)

        # Probability of class y given data x.
        self.p_y_given_x = T.nnet.softmax(
            T.dot(input_data, self.weights) + self.biases)

        # Get class with highest probability.
        self.y_prediction = T.argmax(
            self.p_y_given_x, axis=1)

        # Parameters of the model.
        self.params = [self.weights, self.biases]

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

