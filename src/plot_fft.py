#!/usr/bin/ipython
"""Plots the FFT of a wav file."""

import pylab
from pylab import *
import wav_file_importer

sample_frequency, data = wav_file_importer.validate_and_read_file()

number_of_samples = len(data)
fft_output = fft(data)

nUniqueParts = ceil((number_of_samples + 1)/2.0)

fft_output = fft_output[0:nUniqueParts]
fft_output = abs(fft_output)

fft_output = fft_output / float(number_of_samples)

fft_output = fft_output**2

if number_of_samples % 2 > 0:
    fft_output[1:len(fft_output)] = fft_output[1:len(fft_output)]*2
else:
    fft_output[1:len(fft_output) - 1] = fft_output[1:len(fft_output) - 1] * 2

freqArray = arange(0, nUniqueParts, 1.0) * (sample_frequency / number_of_samples)

plot(freqArray/1000, 10*log10(fft_output), color='k')
xlabel('Frequency (kHz)')
ylabel('Power (dB)')
pylab.show()
