"""
Class to hold the compiled theano functions
for logistic modeling.
"""

import theano
import theano.tensor as T
import birdbot.params as p

#pylint: disable=R0903,C0103,E0602

class Functions(object):
    """Compiled functions for logistic regression."""

    def __init__(self, data, symbolic_vars, classifier):
        """Set up our functions."""

        # Break out the symbolic variables.
        index, x, y = symbolic_vars

        # Set up the cost function.
        cost = classifier.negative_log_likelihood(T.cast(y, 'int64'))

        # compute the gradient of cost with respect to theta = (W,b)
        g_weights = T.grad(cost=cost, wrt=classifier.weights)
        g_biases = T.grad(cost=cost, wrt=classifier.biases)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(classifier.weights,
                    classifier.weights - p.LEARNING_RATE * g_weights),
                   (classifier.biases,
                    classifier.biases - p.LEARNING_RATE * g_biases)]

        # Set up the test function.
        self.test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(T.cast(y, 'int64')),
            givens={
                x: data.shared_test_x[
                    index *p.BATCH_SIZE: (index + 1) *p.BATCH_SIZE],
                y: data.shared_test_y[
                    index * p.BATCH_SIZE: (index + 1) * p.BATCH_SIZE]})

        # Set up the validation function.
        self.validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(T.cast(y, 'int64')),
            givens={
                x: data.shared_valid_x[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE],
                y: data.shared_valid_y[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE]})

        # Set up training function.
        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: data.shared_train_x[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE],
                y: data.shared_train_y[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE]})

