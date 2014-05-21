#!/usr/bin/ipython

import pylab
from pylab import plot, ylabel, xlabel, arange
import wav_file_importer

sample_frequency, data = wav_file_importer.validate_and_read_file()

number_of_samples = data.shape[0]

# Create the X-Axis.
time_array = arange(0, float(number_of_samples), 1)

# Convert to the correct actual times.
time_array = time_array / sample_frequency

plot(time_array, data, color='k')
ylabel("Amplitude")
xlabel("Time (s)")
pylab.show()

