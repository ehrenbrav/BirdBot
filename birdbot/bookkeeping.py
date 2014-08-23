"""Global stats for the model."""

import numpy as np
import time
import birdbot.params as p

class Bookkeeping(object):
    """Keep track of global stats."""

    #pylint: disable=R0903

    def __init__(self):
        """Set up metrics."""

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
        print(('Optimization complete: Best validation score of %f %%,'
        'with test performance %f %%') %
        (self.best_validation_loss * 100., self.test_score * 100.))

        print 'The code run for %d epochs, with %f epochs/sec' % (
            self.epoch, 1. * self.epoch / (end_time - self.start_time))

        print "Total Time: %.1fs" % (end_time - self.start_time)

