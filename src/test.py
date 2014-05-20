#!/usr/bin/python
"""Test class for trying out FFTs"""
from pylab import *
from scipy.io import wavfile

sample_frequency, data  = wavfile.read('../crows.wav')
number_of_samples = data.shape[0]

print("Sample Rate: " + str(sample_frequency))
print("Data Type: " + str(data.dtype))
print("Length (seconds): " + str(number_of_samples / sample_frequency))

time_array = arange(0, number_of_samples, 1)
time_array = time_array / sample_frequency
time_array = time_array * 1000

plot(time_array, data, color = 'k')
