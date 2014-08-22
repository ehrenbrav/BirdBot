"""
Load the dataset, which must be
cPickled, gzipped, and in a
particular format.
"""

import cPickle
import gzip
import batch_logistic as bl

# Path to the pickled, gzipped data.
DATASET_PATH = 'dataset.pkl.gz'

def load():
    """Returns parsed data, ready for use."""
    
    # Import the data from file.
    print "Loading data..."
    datafile = gzip.open(DATASET_PATH, 'rb')
    train_set, valid_set, test_set, classification_map = cPickle.load(datafile)
    datafile.close()

    train_set_list = split_data(train_set)
    valid_set_list = split_data(valid_set)
    test_set_list = split_data(test_set)

    return train_set_list, valid_set_list, test_set_list, classification_map

def split_data(dataset):
    """
    Divide the data into lists, where each member has a maximum number
    of examples.
    """
    return_list = []
    counter = 0
    for data_element in enumerate(dataset):

        # Initialize a new element.
        return_list_element = []

        # Populate this element until full.
        while counter < bl.MAX_DATA_SIZE:
            return_list_element.append(data_element)

        return_list.append(return_list_element)
        counter = 0

    return return_list


    
