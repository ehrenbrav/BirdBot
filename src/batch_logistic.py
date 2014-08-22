#!/usr/bin/env python

import theano
import theano.tensor as T

import numpy as np

import load_data as ld
import logistic_classifier as lc

# Set the maximum number of examples to copy to the GPU at one time.
MAX_DATA_SIZE = 10000

def sgd_optimization(learning_rate=.13, n_epochs=50, batch_size=50):
    """Use stochastic gradient descent to optimize a prediction model."""

    # Read the data into memory.
    print "Building model..."
    datasets = ld.load()
    train_set_list = datasets[0]
    valid_set_list = datasets[1]
    test_set_list = datasets[2]
    classification_map = datasets[3]

    # Allocate our symbolic data variables.
    index = T.scalar()   # Minibatch index.
    x = T.matrix('x')    # Image data.
    y = T.vector('y')    # Classifications

    # Allocate the shared variables and initialize.
    shared_x = theano.shared(np.asarray(
        train_set_list[0][0], dtype=theano.config.floatX), borrow=True)
    shared_y_float = theano.shared(np.asarray(
        train_set_list[0][1], dtype=theano.config.floatX), borrow=True)
    shared_y = T.cast(shared_y_float, 'int32')

    # Calculate the size of our images.
    num_pixels = train_set_x.get_value().shape[1]

    # Calculate the number of different classifications.
    num_classifications = len(classification_map)

    # construct the logistic regression class
    classifier = lc.LogisticRegression(
        input=x, n_in=num_pixels, n_out=num_classifications)

    # Set up the cost function.
    cost = classifier.negative_log_likelihood(y)
    
if __name__ == '__main__':
    sgd_optimization()
        










