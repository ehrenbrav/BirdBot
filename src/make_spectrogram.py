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
import os
import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
from matplotlib import pyplot

# How long do we want our spectrograms?
SAMPLE_DURATION = 4

# Where to save the spectrograms.
SPECTROGRAM_SAVE_PATH = "../training_data/spectrograms/"

def graph_spectrogram(path):
    """Create the spectrograms and save as png files."""

    # Get the name of the wav file.
    filename = os.path.basename(path)
    print "Processing " + filename

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
    for counter, sample in enumerate(front_samples + end_samples):

        # Get add the number of spectrogram sample this is.
        savename = filename
        savename = savename.replace(".wav", "_" + str(counter) + ".png")
        savename = savename.replace(".mp3", "_" + str(counter) + ".png")
        savepath = SPECTROGRAM_SAVE_PATH + savename

        # If the spectrogram exists, continue.
        if os.path.exists(savepath):
            continue

        pyplot.gray()
        pyplot.specgram(sample, Fs=sample_rate)

        # Limit the frequencies.
        pyplot.ylim((100, 13000))

        # Save.
        pyplot.savefig(savepath, bbox_inches='tight', pad_inches=0)
        print "Writing " + savepath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    path = parser.parse_args().file_path

    # If the argument is a file, make the spectrogram.
    if os.path.isfile(path):
        graph_spectrogram(path)

    else:
        for file_in_dir in os.listdir(path):
            file_in_dir_path = os.path.join(path, file_in_dir)
            graph_spectrogram(file_in_dir_path)
