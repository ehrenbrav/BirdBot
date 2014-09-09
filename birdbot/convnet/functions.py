"""
Class to hold the compiled theano functions
for convnet modeling.
"""

import theano
import theano.tensor as T
import birdbot.params as p

#pylint: disable=R0903,C0103,E0602

class Functions(object):
    """Compiled functions for the convnet."""

    def __init__(self, data, symbolic_vars, classifier, params):
        """Set up our functions."""

        # Break out the symbolic variables.
        index, x, y = symbolic_vars

        # Set up the cost function.
        cost = classifier.negative_log_likelihood(T.cast(y, 'int64'))

        # Create a list of gradients for all parameters.
        grads = T.grad(cost, params)

        # This updates list is created by looping over all (params[i],
        # grads[i]) pairs.
        updates = []
        for param_i, grad_i in zip(params, grads):

            # Update to the next iteration.
            update = (-p.LEARNING_RATE * grad_i) + \
                (-p.WEIGHT_DECAY * p.LEARNING_RATE * param_i)

            updates.append((param_i, param_i + update))

        # Set up the test function.
        self.test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(T.cast(y, 'int64')),
            name='test_model',
            givens={
                x: data.shared_test_x[
                    index * p.BATCH_SIZE: (index + 1) * p.BATCH_SIZE],
                y: data.shared_test_y[
                    index * p.BATCH_SIZE: (index + 1) * p.BATCH_SIZE]})

        # Set up the validation function.
        self.validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(T.cast(y, 'int64')),
            name='validate_model',
            givens={
                x: data.shared_valid_x[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE],
                y: data.shared_valid_y[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE]})

        # Set up training function.
        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            name='train_model',
            updates=updates,
            givens={
                x: data.shared_train_x[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE],
                y: data.shared_train_y[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE]})

        # Set up function for testing the training model.
        self.test_training_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(T.cast(y, 'int64')),
            name='test_training_model',
            givens={
                x: data.shared_train_x[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE],
                y: data.shared_train_y[
                    index * p.BATCH_SIZE:(index + 1) * p.BATCH_SIZE]})

