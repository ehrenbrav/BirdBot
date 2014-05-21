#!/usr/bin/ipython

import pylab
from pylab import *
import wav_file_importer

sample_frequency, data = wav_file_importer.validate_and_read_file()

n = len(data)
p = fft(data)

nUniqueParts = ceil((n+1)/2.0)

p = p[0:nUniqueParts]
p = abs(p)

p = p / float(n)

p = p**2

if n % 2 > 0:
    p[1:len(p)] = p[1:len(p)]*2
else:
    p[1:len(p) - 1] = p[1:len(p) - 1] * 2

freqArray = arange(0, nUniqueParts, 1.0) * (sample_frequency / n)

plot(freqArray/1000, 10*log10(p), color='k')
xlabel('Frequency (kHz)')
ylabel('Power (dB)')
pylab.show()
