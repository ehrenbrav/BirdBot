"""This module generates the features for each frame it is passed."""

import libxtract.xtract as xtract
import scipy as sp
from scipy import signal, fftpack
import pylab as plt
import numpy as np

def extract_features(frame):
    """Extract the features from a given frame."""
    number_of_samples = len(frame.samples)
    spectrum = calculate_fft(calculate_windowed_frame(frame.samples, number_of_samples))
    amplitude_data = xtract.doubleArray(number_of_samples)
    spectral_data = xtract.doubleArray(number_of_samples)
    for i in range(len(frame.samples)):
        amplitude_data[i] = int(frame.samples[i])
        spectral_data[i] = int(spectrum[i])
    mean = calculate_mean(amplitude_data, number_of_samples)
    argv = xtract.doubleArray(1)
    argv[0] = mean
    print "Mean: " + str(mean)
    print "Variance: " \
        + str(calculate_variance(amplitude_data, number_of_samples, argv))
    print "Spectral Centroid: " \
        + str(calculate_spectral_centroid(spectral_data, number_of_samples))
    print "Spectral Variance: " \
        + str(calculate_spectral_variance(
            spectral_data, number_of_samples, argv))
    print "Spectral Rolloff: " \
        + str(calculate_rolloff(spectral_data, number_of_samples))
    time = np.asarray(range(512))
    samples = np.asarray(frame.samples)
    plt.subplot(3, 1, 1)
    plt.plot(time, samples)
    plt.subplot(3, 1, 2)
    plt.subplot(3, 1, 3)
    plt.plot(time, spectrum)
    plt.show()

def calculate_mean(amplitude_data, number_of_samples):
    """Calculate the mean of a frame."""
    return xtract.xtract_mean(amplitude_data, number_of_samples, None)[1]

def calculate_variance(amplitude_data, number_of_samples, argv):
    """Calculate the variance of a frame."""
    return xtract.xtract_variance(amplitude_data, number_of_samples, argv)[1]

def calculate_windowed_frame(samples, number_of_samples):
    """Return the frame after applying
    a Hamming window.."""
    return samples * sp.signal.hamming(number_of_samples)

def calculate_fft(samples):
    """Calculate the FFT."""
    return abs(fftpack.rfft(samples))

def calculate_spectral_centroid(spectral_data, number_of_samples):
    """Calculate the spectral centroid."""
    return xtract.xtract_spectral_centroid(spectral_data, number_of_samples, None)[1]

def calculate_spectral_variance(spectral_data, number_of_samples, argv):
    """Calculate the bandwidth."""
    return xtract.xtract_spectral_variance(spectral_data, number_of_samples, argv)[1]

def calculate_rolloff(spectral_data, number_of_samples):
    """Calculate spectral rolloff at 95%."""
    argv = xtract.doubleArray(2)
    argv[0] = float(44100) / float(number_of_samples)
    argv[1] = 0.95
    return xtract.xtract_rolloff(spectral_data, number_of_samples, argv)[1]

