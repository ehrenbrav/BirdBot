#!/usr/bin/env python
"""
This module divides a wav file into euqally sized
spectrograms. In order to increase the amount of training
data available, it first starts from the beginning of the file,
discarding the remainder that doesn't divide equally into the
spectrogram frame size, then starts with an offset equal to the
remainder, to ensure that the entire wav file is covered.
"""

import os
import wave
import argparse
import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
import pylab
import numpy as np

GRAPH_WIDTH = 1024
GRAPH_HEIGHT = 128

def graph_spectrogram(wav_file):
    data, frame_rate = get_wav_info(wav_file)
    Pxx, freqs, time, im = pylab.specgram(data, Fs=frame_rate)
    remainder = Pxx.shape[1] % GRAPH_WIDTH
    first_section = Pxx[:, 0:Pxx.shape[1] - remainder]
    pieces = np.split(first_section, first_section.shape[1]/GRAPH_WIDTH, axis=1)
    counter = 0
    for piece in pieces:
        pylab.plot(piece[0], piece[1])
        pylab.savefig('spectrogram' + str(counter) + '.png')
        counter += 1

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()

    if frame_rate != 44100:
        print "Error: frame rate is not 44100."
        exit(1)
    return sound_info, frame_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    file_path = parser.parse_args().file_path
    graph_spectrogram(file_path)
