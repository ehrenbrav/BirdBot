#!/usr/bin/env python
"""
Train a convolutional network.
"""

#pylint: disable=C0103

# Built-ins
import logging
import inspect
import argparse

# Packages.
import theano
import theano.tensor as T
import numpy as np

# BirdBot.
from birdbot import data_handler, bookkeeping, number_crunching, fileIO, logger
import birdbot.params as p
import birdbot.logistic.classifier as lc
import birdbot.convnet.functions as cf
import birdbot.convnet.pool_conv_layer as pcl
from birdbot.convnet.mlp import HiddenLayer

def train_convnet(
        n_kerns, filter_size, poolsize, logistic_inputs, saved_model=None):
    """Use stochastic gradient descent to optimize the convnet model."""

    # Set up logging.
    logger.Logger()
    call_values = {"n_kerns": n_kerns,
                   "filter_size": filter_size,
                   "poolsize": poolsize,
                   "logistic_inputs": logistic_inputs}
    logging.debug("--------------------------")
    if saved_model == None:
        logging.debug("Starting new run of convnet.")
    else:
        logging.debug("Resuming training of convnet.")

    # Load saved data, if necessary.
    # init_params is a list of W and b for each layer.
    init_params = [None, None, None, None]
    saved = None
    if saved_model != None:
        saved = fileIO.FileLoader(saved_model)
        init_params = saved.init_params

    # Set up bookeeping and logging.
    if saved_model == None:
        bk = bookkeeping.Bookkeeping()
    else:
        bk = saved.bk

    # Log initial params.
    logging.debug(
        "train_convnet params: " + str(call_values))
    logging.debug(print_hyperparams())

    # Set up and load data.
    data = data_handler.DataHandler()

    # Initialize symbolic variables.
    logging.info("Building model...")
    index = T.scalar('index', dtype='int64')      # Minibatch index.
    x = T.matrix('x', dtype=theano.config.floatX) # Image data.
    y = T.vector('y', dtype=theano.config.floatX) # Classifications
    symbolic_variables = (index, x, y)

    # Reshape the initial layer.
    spectrogram_side = int(np.sqrt(data.shared_train_x.get_value().shape[1]))
    layer0_input = x.reshape(
        (p.BATCH_SIZE,
         p.PIXEL_DIM,
         spectrogram_side,
         spectrogram_side))

    # Create the first convolutional layer.
    layer0_image_shape = (
        p.BATCH_SIZE,
        p.PIXEL_DIM,
        spectrogram_side,
        spectrogram_side)
    layer0_filter_shape = (
        n_kerns[0],
        p.PIXEL_DIM,
        filter_size[0],
        filter_size[0])
    layer0 = pcl.LeNetConvPoolLayer(
        data_input=layer0_input,
        image_shape=layer0_image_shape,
        filter_shape=layer0_filter_shape,
        poolsize=poolsize,
        init_params=init_params[0])

    # Wire the first layer to the second.
    layer1_input_dim = \
    (spectrogram_side - filter_size[0] + 1) / poolsize[0]

    # Create the second convolutional layer.
    layer1 = pcl.LeNetConvPoolLayer(
        data_input=layer0.output,
        image_shape=(
            p.BATCH_SIZE,
            n_kerns[0],
            layer1_input_dim,
            layer1_input_dim),
        filter_shape=(
            n_kerns[1],
            n_kerns[0],
            filter_size[1],
            filter_size[1]),
        poolsize=poolsize,
        init_params=init_params[1])

    # Wire layer1 to the fully-connected sigmoidal layer.
    layer2_input_dim = \
    (layer1_input_dim - filter_size[1] + 1) / poolsize[1]

    # Create the fully-connected sigmoidal layer.
    layer2 = HiddenLayer(
        data_input=layer1.output.flatten(2),
        n_in=(n_kerns[1] * layer2_input_dim**2),
        n_out=logistic_inputs,
        activation=T.tanh,
        init_params=init_params[2])

    # Create the logistic layer3.
    layer3 = lc.LogisticClassifier(
        input_data=layer2.output,
        n_in=logistic_inputs,
        n_out=len(data.classification_map),
        init_params=init_params[3])

    # Group all our parameters to optimize into a list.
    params = layer0.params + layer1.params + layer2.params + layer3.params

    # Set up our train, test, validate functions.
    functions = cf.Functions(data, symbolic_variables, layer3, params)

    # Log.
    logging.info("Commencing training...")

    # Start cranking.
    while bk.epoch < p.NUM_EPOCHS:

        # Loop through all of our training data.
        bk.epoch += 1

        # Do the hard work.
        number_crunching.run_calculation(data, functions, bk)

        # Quit if we're out of patience.
        if bk.patience <= bk.iteration:
            break

        # If the save flag is true, save since it's a new high score.
        if bk.SAVE_FLAG:
            init_params = \
              [[layer0.params[0].get_value(), layer0.params[1].get_value()],
               [layer1.params[0].get_value(), layer1.params[1].get_value()],
               [layer2.params[0].get_value(), layer2.params[1].get_value()],
               [layer3.params[0].get_value(), layer3.params[1].get_value()]]
            fileIO.save_model(bk, init_params)
            bk.SAVE_FLAG = False
            
    # Print the final tally.
    bk.print_results()

def print_hyperparams():
    """Returns a string of parameters in an object."""

    params = {}
    for name in dir(p):
        value = getattr(p, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            params[name] = value
    return params

if __name__ == '__main__':

    # See if there's a saved file we're starting from.
    saved_model_path = None
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    args = parser.parse_args()
    if args.f:
        saved_model_path = args.f

    # Get to it.
    # Note that n_kerns must be a multiple of 16.
    train_convnet(
        n_kerns=[32, 64],
        filter_size=[15, 10],
        poolsize=(2, 2),
        logistic_inputs=500,
        saved_model=saved_model_path)
