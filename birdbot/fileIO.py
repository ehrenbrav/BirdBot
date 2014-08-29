"""Handle file writing operations."""

import datetime
import cPickle
import logging
import gzip

def save_model(bk, p, functions):
    
    """Save data so we can pick up later."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
    filename = "../" + timestamp + ".sav.gz"
    output = gzip.open(filename, "wb")
    cPickle.dump(bk, output, -1)
    output.close()
    logging.info("Wrote to " + filename)
    
