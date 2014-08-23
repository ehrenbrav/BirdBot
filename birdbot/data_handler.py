"""
This class loads the dataset, which must be
cPickled, gzipped, and in a
particular format. It also stores
the data for access by other modules.
"""

import cPickle
import gzip
import numpy as np
from params import DATASET_PATH, MAX_DATA_SIZE
import theano

#pylint: disable=R0903,R0902

class DataHandler(object):
    """Load and store the data."""

    def __init__(self):
        """Load data."""

        # Import the data from file.
        print "Loading data..."
        datafile = gzip.open(DATASET_PATH, 'rb')
        _train_set, _valid_set, _test_set, self.classification_map = \
          cPickle.load(datafile)
        datafile.close()

        # Split the data into chunks that will fit on the GPU.
        self.train_set_list = __split_data__(_train_set)
        self.valid_set_list = __split_data__(_valid_set)
        self.test_set_list = __split_data__(_test_set)

        # Allocate the shared variables and initialize.
        self.shared_train_x, self.shared_train_y = __init_shared__(
            self.train_set_list)
        self.shared_valid_x, self.shared_valid_y = __init_shared__(
            self.valid_set_list)
        self.shared_test_x, self.shared_test_y = __init_shared__(
            self.test_set_list)
        
        # Calculate the size of our images.
        self.num_pixels = self.train_set_list[0][0].shape[1]

        # Calculate the number of different classifications.
        self.num_classifications = len(self.classification_map)

def __split_data__(dataset):
    """
    The dataset is a tuple of ndarrays: (data, classification)
    with shape ((n_examples, n_pixels), n_examples)    
    """

    # Calculate how many pieces to split the data into.
    n_chunks = dataset[0].shape[0] / MAX_DATA_SIZE

    # Add one so we get the fractional chunk.
    n_chunks += 1

    # Return a list of sub-arrays.
    x_data = np.array_split(dataset[0], n_chunks)
    y_data = np.array_split(dataset[1], n_chunks)
    
    return zip(x_data, y_data)
    
def __init_shared__(dataset_list):
    """Create shared variables to be used by the GPU."""

    #pylint: disable=E1101
    shared_x = theano.shared(np.asarray(
        dataset_list[0][0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(
        dataset_list[0][1], dtype=theano.config.floatX), borrow=True)
    return shared_x, shared_y




    
