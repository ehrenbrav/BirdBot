"""
This class loads the dataset from the database.
"""

import numpy as np
import logging
import theano
import psycopg2
from birdbot.params import DATABASE, MAX_DATA_SIZE

#pylint: disable=R0903,R0902

class DataHandler(object):
    """Load and store the data."""

    def __init__(self):
        """Load data."""

        # Import the data from DB.
        logging.info("Loading data from database...")

        # Set up the DB connection.
        connection = None
        try:
            connection = psycopg2.connect(database=DATABASE, user='ehrenbrav')
            cursor = connection.cursor()

            # Now, copy the dataset into memory.
            logging.info("Loading testing data...")
            cursor.execute(
                """SELECT data, classification_id FROM spectrograms
                WHERE dataset_category='test';""")
            _test_set = cursor.fetchall()

            logging.info("Loading validation data...")
            cursor.execute(
                """SELECT data, classification_id FROM spectrograms
                WHERE dataset_category='valid';""")
            _valid_set = cursor.fetchall()

            logging.info("Loading training data...")
            cursor.execute(
                """SELECT data, classification_id FROM spectrograms
                WHERE dataset_category='train';""")
            _train_set = cursor.fetchall()

            # Generate the classification map of
            # classification -> classification_id.
            cursor.execute("""SELECT DISTINCT classification, classification_id
            FROM spectrograms;""")

            self.classification_map = cursor.fetchall()

        except psycopg2.DatabaseError, exception:
            if connection:
                connection.rollback()
            logging.error(exception)
            exit(1)

        finally:
            if connection:
                connection.close()

        # Log some statistics.
        logging.info("Number of examples: " + str(
            len(_train_set) +
            len(_test_set) +
            len(_valid_set)))

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

def __split_data__(dataset_list):
    """
    The dataset is a tuple of ndarrays: (data, classification)
    with shape ((n_examples, n_pixels), n_examples)
    """

    # Convert the list to a tupel of ndarrays.
    x_list, y_list = zip(*dataset_list)
    dataset = (np.asarray(x_list), np.asarray(y_list))

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
    # Load the first element of data into the shared memory.
    # If there's only one element in total, we won't touch
    # the shared variables again.
    shared_x = theano.shared(np.asarray(
        dataset_list[0][0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(
        dataset_list[0][1], dtype=theano.config.floatX), borrow=True)
    return shared_x, shared_y

