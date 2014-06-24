#!/usr/bin/ipython
"""Plots the FFT of a wav file."""

# pylint: disable=C0103

import pylab
import numpy as np
import wav_file_importer

sample_frequency, data = wav_file_importer.validate_and_read_file()

number_of_samples = len(data)
fft_output = pylab.fft(data)

nUniqueParts = pylab.ceil((number_of_samples + 1)/2.0)

fft_output = fft_output[0:nUniqueParts]
fft_output = abs(fft_output)

fft_output = fft_output / float(number_of_samples)

fft_output = fft_output**2

if number_of_samples % 2 > 0:
    fft_output[1:len(fft_output)] = fft_output[1:len(fft_output)]*2
else:
    fft_output[1:len(fft_output) - 1] = fft_output[1:len(fft_output) - 1] * 2

freqArray = \
np.arange(0, nUniqueParts, 1.0) * (sample_frequency / number_of_samples)

pylab.plot(freqArray/1000, 10*pylab.log10(fft_output), color='k')
pylab.xlabel('Frequency (kHz)')
pylab.ylabel('Power (dB)')
pylab.show()
