"""
Class for a convolutional pooling layer.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

# pylint: disable=R0903,C0103

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, data_input, filter_shape, image_shape, poolsize):

        assert image_shape[1] == filter_shape[1]
        self.input = data_input

        # Initialize random number generator.
        rng = np.random.RandomState(23455)

        # Initialize weights
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        # Initialize shared model weights.
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # Initialize shared model biases.
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # Convolve data_input feature maps with filters
        conv_out = conv.conv2d(
            input=data_input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape)

        # Downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=poolsize, ignore_border=True)

        # Add the bias term.
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store parameters of this layer
        self.params = [self.W, self.b]
