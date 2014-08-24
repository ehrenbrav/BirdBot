#!/usr/bin/env python
"""
Train a convolutional network.
"""

#pylint: disable=C0103

# Packages.
import theano
import theano.tensor as T
import numpy as np

# BirdBot.
from birdbot import data_handler, bookkeeping, number_crunching
import birdbot.params as p
import birdbot.logistic.classifier as lc
import birdbot.convnet.functions as cf
import birdbot.convnet.pool_conv_layer as pcl
from birdbot.convnet.mlp import HiddenLayer

def train_convnet(n_kerns, filter_size, poolsize, logistic_layer_inputs):
    """Use stochastic gradient descent to optimize the convnet model."""

    # Set up data.
    data = data_handler.DataHandler()

    # Initialize symbolic variables.
    index = T.scalar('index', dtype='int64')      # Minibatch index.
    x = T.matrix('x', dtype=theano.config.floatX) # Image data.
    y = T.vector('y', dtype=theano.config.floatX) # Classifications
    symbolic_variables = (index, x, y)
    
    # Set up the logistic classifier.
    print "Building model..."

    # Reshape the initial layer.
    spectrogram_side = int(np.sqrt(data.shared_train_x.get_value().shape[1]))
    layer1_input = x.reshape(
        (p.BATCH_SIZE,
         p.PIXEL_DIM,
         spectrogram_side,
         spectrogram_side))

    # Create the first convolutional layer.
    layer1_image_shape = (
        p.BATCH_SIZE,
        p.PIXEL_DIM,
        spectrogram_side,
        spectrogram_side)
    layer1_filter_shape = (
        n_kerns[0],
        p.PIXEL_DIM,
        filter_size[0],
        filter_size[0])
    layer1 = pcl.LeNetConvPoolLayer(
        data_input=layer1_input,
        image_shape=layer1_image_shape,
        filter_shape=layer1_filter_shape,
        poolsize=poolsize)

    # Wire the first layer to the second.
    layer2_input_dim = \
    (spectrogram_side - filter_size[0] + 1) / poolsize[0]

    # Create the second convolutional layer.
    layer2 = pcl.LeNetConvPoolLayer(
        data_input=layer1.output,
        image_shape=(
            p.BATCH_SIZE,
            n_kerns[0],
            layer2_input_dim,
            layer2_input_dim),
        filter_shape=(
            n_kerns[1],
            n_kerns[0],
            filter_size[1],
            filter_size[1]),
        poolsize=poolsize)
    
    # Wire layer2 to the fully-connected sigmoidal layer.
    layer3_input_dim = \
    (layer2_input_dim - filter_size[1] + 1) / poolsize[1]

    # Create the fully-connected sigmoidal layer.
    layer3 = HiddenLayer(
        data_input=layer2.output.flatten(2),
        n_in=(n_kerns[1] * layer3_input_dim**2),
        n_out=logistic_layer_inputs,
        activation=T.tanh)

    # Create the logistic classifier.
    classifier = lc.LogisticClassifier(
        input_data=layer3.output,
        n_in=logistic_layer_inputs,
        n_out=len(data.classification_map))

    # Group all our parameters to optimize into a list.
    params = layer1.params + layer2.params + layer3.params + classifier.params
    
    # Set up our train, test, validate functions.
    functions = cf.Functions(data, symbolic_variables, classifier, params)

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
    train_convnet(
        n_kerns=[32, 64],
        filter_size=[15, 10],
        poolsize=(2, 2),
        logistic_layer_inputs=500)
