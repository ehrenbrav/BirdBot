#!/usr/bin/env python
"""
Perform logistic regression to train a
model on the dataset.
"""

#pylint: disable=C0103

# Built-ins.
import time

# Packages.
import theano
import theano.tensor as T
import numpy as np

# BirdBot.
import birdbot.data_handler as dh
import birdbot.logistic.classifier as lc
import birdbot.logistic.functions as lf
import birdbot.params as p

class Bookkeeping(object):
    """Keep track of global stats."""

    #pylint: disable=R0903
    
    def __init__(self):
        """Set up metrics."""
        
        self.best_validation_loss = np.inf
        self.test_score = 0.
        self.start_time = time.clock()
        self.patience = p.PATIENCE
        self.epoch = 0
        self.iteration = 0
        
    def print_results(self):
        """Print a summary of the results."""
    
        end_time = time.clock()
        print(('Optimization complete: Best validation score of %f %%,'
        'with test performance %f %%') %
        (self.best_validation_loss * 100., self.test_score * 100.))
    
        print 'The code run for %d epochs, with %f epochs/sec' % (
            self.epoch, 1. * self.epoch / (end_time - self.start_time))

        print "Total Time: %.1fs" % (end_time - self.start_time)

def train_logistic_model():
    """Use stochastic gradient descent to optimize a prediction model."""

    # pylint: disable=E0602

    # Set up data.
    data = dh.DataHandler()

    # Initialize symbolic variables.
    index = T.scalar('index', dtype='int64')      # Minibatch index.
    x = T.matrix('x', dtype=theano.config.floatX) # Image data.
    y = T.vector('y', dtype=theano.config.floatX) # Classifications

    # Set up the logistic classifier.
    print "Building model..."
    classifier = lc.LogisticClassifier(
        input_data=x, n_in=data.num_pixels, n_out=data.num_classifications)

    # Set up our train, test, validate functions.
    functions = lf.Functions(data, (index, x, y), classifier)

    # Set up bookeeping.
    bk = Bookkeeping()

    # Start cranking.
    while bk.epoch < p.NUM_EPOCHS:

        # Loop through all of our training data.
        bk.epoch += 1

        # Do the hard work.
        run_calculation(data, functions, bk)

        # Quit if we're out of patience.
        if bk.patience <= bk.iteration:
            break

    # Summarize the results
    bk.print_results()
        
def run_calculation(data, functions, bk):
    """Do the actual work."""

    # Loop through the chunks of data.
    for x, y in data.train_set_list:

        # Load this bit of data into our shared variables.
        data.shared_train_x.set_value(x)
        data.shared_train_y.set_value(y)

        # Calculate the number of minibatches to run through.
        n_batches = data.shared_train_x.get_value().shape[0] / p.BATCH_SIZE

        # Loop through the minibatches.
        for minibatch_index in xrange(n_batches):

            # Train the model.
            functions.train_model(minibatch_index)

            # Increment the iteration number.
            bk.iteration += 1

            # If the time is right, calculate validation and/or test scores.
            if (bk.iteration + 1) % p.VALIDATION_FREQUENCY == 0:

                # Run the accuracy calculation.
                compute_accuracy(
                    functions, data, bk, minibatch_index, n_batches)

def compute_accuracy(functions, data, bk, minibatch_index, n_train_batches):
    """
    Run the model on the validation
    and testing sets (if applicable).
    """

    # Calculate the validation score.
    validation_losses = test_model(
        data.valid_set_list,
        data.shared_valid_x,
        data.shared_valid_y,
        functions.validate_model)
    this_validation_loss = np.mean(validation_losses)

    print('Epoch %i, minibatch %i/%i, validation error %f %%' %\
          (bk.epoch, minibatch_index + 1, n_train_batches,
           this_validation_loss * 100))

    # If we have the best validation score so far...
    if this_validation_loss < bk.best_validation_loss:

        # Improve patience if the gain is big enough.
        if this_validation_loss < bk.best_validation_loss * \
          p.IMPROVEMENT_THRESHOLD:
                      
            bk.patience = max(
                bk.patience, bk.iteration * p.PATIENCE_INCREASE)
            
            bk.best_validation_loss = this_validation_loss

            # Try the test set.
            test_losses = test_model(
                data.test_set_list,
                data.shared_test_x,
                data.shared_test_y,
                functions.test_model)
            bk.test_score = np.mean(test_losses)

            print(('  Epoch %i, minibatch %i/%i, '
                  'test error of best '
                  'model %f %%')
                  % (bk.epoch,
                  minibatch_index + 1,
                  n_train_batches,
                  bk.test_score * 100.))

def test_model(data_list, shared_x, shared_y, function):
    """Run the model on the validation or testing set."""

    # Loop through all the data chunks in our set.
    losses = []
    for x, y in data_list:
        
        # Update the shared variables.
        shared_x.set_value(x)
        shared_y.set_value(y)

        # Calculate the number of minibatches we need.
        n_batches = shared_x.get_value().shape[0] / p.BATCH_SIZE

        # Loop through the minibatches.
        for minibatch in xrange(n_batches):

            # Run the calculation.
            losses.append(function(minibatch))
        
    return losses

if __name__ == '__main__':
    train_logistic_model()
