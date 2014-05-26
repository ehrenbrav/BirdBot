"""
This is the class for an audio frame.
Typically it is 512 samples.
"""
import pylab as plt
import numpy as np

class Frame():

    FRAME_SIZE = 512

    def __init__(self):
        self.samples = []
        self.windowed_samples = []
        self.spectrum = []
        self.mean = 0
        self.variance = 0
        self.spectral_centroid = 0
        self.spectral_variance = 0
        self.spectral_rolloff = 0

    def print_frame_stats(self):
        """Print some basic debugging stats."""
        print "Mean: " + str(self.mean)
        print "Variance: " + str(self.variance)
        print "Spectral Centroid: " + str(self.spectral_centroid)
        print "Spectral Variance: " + str(self.spectral_variance)
        print "Spectral Rolloff: " + str(self.spectral_rolloff)

    def graph_frame(self):
        """Make a graph for debugging purposes."""
        time = np.asarray(range(len(self.samples)))
        samples = np.asarray(self.samples)
        plt.subplot(3, 1, 1)
        plt.plot(time, samples)
        plt.subplot(3, 1, 2)
        plt.plot(time, self.windowed_samples)
        plt.subplot(3, 1, 3)
        plt.plot(time, self.spectrum)
        plt.show()
        
