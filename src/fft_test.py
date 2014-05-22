#!/usr/bin/env python
"""Tests the fft functions of numpy."""

from scipy import fftpack, pi, linspace, sin, random, fft, log10
import pylab

t = linspace(0, 120, 4000)

def acc(t):
    
    return 10*sin(2*pi*2.0*t) + 5*sin(2*pi*8.0*t) + 2*random.random(len(t))

signal = acc(t)

FFT = abs(fft(signal))
frequencies = fftpack.fftfreq(signal.size, t[1] - t[0])

pylab.subplot(211)
pylab.plot(t, signal)
pylab.subplot(212)
pylab.plot(frequencies, 20*log10(FFT), 'x')
pylab.show()
