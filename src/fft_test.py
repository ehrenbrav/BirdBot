#!/usr/bin/env python
"""Tests the fft functions of numpy."""

import scipy as sp
from scipy import fftpack, signal
import pylab

t = sp.linspace(0, 30, 2000)

def acc(t):
    
    return 15*sp.sin(2*sp.pi*2.0*t) + 10*sp.sin(2*sp.pi*8.0*t) + .5*sp.random.random(len(t))

signal = acc(t)
windowed_signal = acc(t) * sp.signal.hamming(len(t))

fft = abs(fftpack.rfft(signal))
windowed_fft = abs(fftpack.rfft(windowed_signal))
frequencies = fftpack.rfftfreq(signal.size, t[1] - t[0])

pylab.subplot(4, 1, 1)
pylab.plot(t, signal)
pylab.subplot(4, 1, 2)
pylab.plot(t, windowed_signal)
pylab.subplot(4, 1, 3)
pylab.plot(frequencies, 20*sp.log10(fft))
pylab.subplot(4, 1, 4)
pylab.plot(frequencies, 20*sp.log10(windowed_fft))
pylab.show()
