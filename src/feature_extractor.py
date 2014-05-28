"""This module generates the features for each frame it is passed."""

import libxtract.xtract as xtract
import scipy as sp
import numpy as np
from scipy import fftpack, signal

def extract_features(frame):
    """Extract the features from a given frame."""

    # Set up the data types for libxtract.
    number_of_samples = len(frame.samples)
    amplitude_data = xtract.doubleArray(number_of_samples)
    spectral_data = xtract.doubleArray(number_of_samples)

    # Create the windowed signal and FFT.
    frame.windowed_samples = calculate_windowed_frame(frame.samples, number_of_samples)
    mag_spectrum = calculate_fft(frame.windowed_samples)
    pow_spectrum = (1.0/number_of_samples) * np.square(mag_spectrum)
    frame.log_pow_spectrum = calculate_log_pow_spectrum(pow_spectrum)

    # Copy this data into double arrays.
    for i in range(len(frame.samples)):
        amplitude_data[i] = int(frame.samples[i])
        spectral_data[i] = int(frame.log_pow_spectrum[i])
        
    # Start storing the data.
    frame.mean = calculate_mean(amplitude_data, number_of_samples)
    argv = xtract.doubleArray(1)
    argv[0] = frame.mean
    frame.variance = calculate_variance(amplitude_data, number_of_samples, argv)
    frame.spectral_centroid = calculate_spectral_centroid(spectral_data, number_of_samples)
    frame.spectral_variance = calculate_spectral_variance(spectral_data, number_of_samples, argv)
    frame.spectral_rolloff = calculate_rolloff(spectral_data, number_of_samples)

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

def calculate_fft(windowed_samples):
    """Calculate the FFT."""
    return abs(fftpack.rfft(windowed_samples))

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

def calculate_log_pow_spectrum(pow_spectrum, normalize=1):
    """Calculate the log of the power spectrum."""
    pow_spectrum[pow_spectrum <= 1e-30] = 1e-30
    return 10 * np.log10(pow_spectrum)

