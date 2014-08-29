# pylint: disable=C0103,R0903
"""Handle file writing operations."""

import datetime
import cPickle
import logging
import gzip

class FileLoader(object):
    """Load and store a saved model."""

    def __init__(self, saved_model):
        """Load the file and parse."""

        logging.info("Loading saved file...")
        input_file = gzip.open(saved_model, "rb")
        self.bk = cPickle.load(input_file)
        self.init_params = cPickle.load(input_file)
        input_file.close()

def save_model(bk, init_params):
    """Save data so we can pick up later."""
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
    filename = "../" + timestamp + ".sav.gz"
    output = gzip.open(filename, "wb")

    # Be extra careful that objects are read in the same
    # order as they're written.
    cPickle.dump(bk, output, -1)
    cPickle.dump(init_params, output, -1)
    output.close()
    logging.info("Wrote to " + filename)
    
