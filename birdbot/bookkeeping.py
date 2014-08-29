"""Global stats for the model."""

import numpy as np
import time
import logging
import birdbot.params as p

class Bookkeeping(object):
    """Keep track of global stats."""

    #pylint: disable=R0903

    def __init__(self):
        """Set up metrics."""

        # Model metrics.
        self.best_validation_loss = np.inf
        self.test_score = 0.
        self.start_time = time.clock()
        self.patience = p.PATIENCE
        self.epoch = 0
        self.iteration = 0
        self.validation_frequency = 50

    def print_results(self):
        """Print a summary of the results."""

        end_time = time.clock()
        summary1 = 'Done: Best validation score: %f %%, test score:  %f %%' % \
        (self.best_validation_loss * 100., self.test_score * 100.)
        logging.debug(summary1)

        summary2 = 'The code run for %d epochs, with %f epochs/sec' % (
            self.epoch, 1. * self.epoch / (end_time - self.start_time))
        logging.debug(summary2)

        summary3 = "Total Time: %.1fs" % (end_time - self.start_time)
        logging.debug(summary3)

