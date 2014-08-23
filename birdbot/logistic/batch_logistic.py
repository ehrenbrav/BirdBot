#!/usr/bin/env python
"""
Perform logistic regression to train a
model on the dataset.
"""

#pylint: disable=C0103

# Packages.
import theano
import theano.tensor as T

# BirdBot.
from birdbot import data_handler, bookkeeping, number_crunching
import birdbot.logistic.classifier as lc
import birdbot.logistic.functions as lf
import birdbot.params as p

def train_logistic_model():
    """Use stochastic gradient descent to optimize a prediction model."""

    # pylint: disable=E0602

    # Set up data.
    data = data_handler.DataHandler()

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
    bk = bookkeeping.Bookkeeping()

    # Start cranking.
    while bk.epoch < p.NUM_EPOCHS:

        # Loop through all of our training data.
        bk.epoch += 1

        # Do the hard work.
        number_crunching.run_calculation(data, functions, bk)

        # Quit if we're out of patience.
        if bk.patience <= bk.iteration:
            break

    # Summarize the results
    bk.print_results()

if __name__ == '__main__':
    train_logistic_model()
