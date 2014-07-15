#!/usr/bin/env python
"""
This module divides a wav file into euqally sized
spectrograms. In order to increase the amount of training
data available, it first starts from the beginning of the file,
discarding the remainder that doesn't divide equally into the
spectrogram frame size, then starts with an offset equal to the
remainder, to ensure that the entire wav file is covered.
"""

#pylint: disable=C0103

import wav_file_importer as wfi
import argparse
import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
from matplotlib import pyplot

SAMPLE_DURATION = 3

def graph_spectrogram(path):
    """
    Create the spectrograms and save as png files.
    """

    # Load the wav file.
    sample_rate, full_audio_data = wfi.validate_and_read_file(path)
    frame_size = sample_rate * SAMPLE_DURATION
    remainder = len(full_audio_data) % frame_size

    # Check to see we actually have enough audio data.
    if len(full_audio_data) < frame_size:
        print "Audio clip is too short"
        exit(1)

    # Chop the wav file into equally sized pieces, starting from
    # the front, then do the same starting from the back.
    front_samples = [full_audio_data[i:i+frame_size] \
            for i in range(0, len(full_audio_data) - remainder, frame_size)]

    end_samples = [full_audio_data[i:i+frame_size] \
            for i in range(remainder, len(full_audio_data), frame_size)]

    # Make the spectrograms.
    counter = 0
    for sample in front_samples + end_samples:

        pyplot.gray()
        pyplot.specgram(sample, Fs=sample_rate)

        # Limit the frequencies.
        pyplot.ylim((100, 13000))

        pyplot.savefig('spectrogram' + str(counter) + '.png',
                       bbox_inches='tight', pad_inches=0)
        counter = counter + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    file_path = parser.parse_args().file_path
    graph_spectrogram(file_path)
