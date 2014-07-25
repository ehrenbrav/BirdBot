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
import logging
import datetime
import argparse
import os
import matplotlib as mpl
import psycopg2
import sys
import cPickle

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')

from matplotlib import pyplot
import numpy as np

# Write the actual spectrogram?
WRITE_SPECTROGRAM=False

# How long do we want our spectrograms?
SPECTROGRAM_DURATION = 4

# Frequency limits.
MAX_FREQUENCY = 13000
MIN_FREQUENCY = 100

def graph_spectrogram(source_path, destination_path, dataset):
    """Create the spectrograms and save as png files."""

    # Get the name of the wav file.
    filename = os.path.basename(source_path)
    logging.info("Processing " + filename)
    print "Processing " + filename

    # Load the wav file.
    sample_rate, full_audio_data = wfi.validate_and_read_file(source_path)
    frame_size = sample_rate * SPECTROGRAM_DURATION
    remainder = len(full_audio_data) % frame_size

    # Check to see we actually have enough audio data.
    if len(full_audio_data) < frame_size:
        logging.error("Audio clip is too short")
        return

    # Get the classification of this recording.
    classification = get_classification(filename)

    # Chop the wav file into equally sized pieces, starting from
    # the front, then do the same starting from the back.
    front_samples = [full_audio_data[i:i+frame_size] \
            for i in range(0, len(full_audio_data) - remainder, frame_size)]

    end_samples = [full_audio_data[i:i+frame_size] \
            for i in range(remainder, len(full_audio_data), frame_size)]
    
    # Make the spectrograms.
    for counter, sample in enumerate(front_samples + end_samples):

        # Compute the spectrogram.
        Pxx, freqs, bins, im  = pyplot.specgram(
            sample, NFFT=1024, Fs=sample_rate, noverlap=512)

        # Chop off useless frequencies.
        Pxx = Pxx[(freqs > MIN_FREQUENCY) & (freqs < MAX_FREQUENCY)]

        # Convert to dB scale and flip.
        data = 10. * np.log10(Pxx)
        data = np.flipud(data)

        # Cast data to int8.
        data = data.astype('int8')

        if WRITE_SPECTROGRAM:
            # Get add the number of spectrogram sample this is.
            savename = filename
            savename = savename.replace(".wav", "_" + str(counter) + ".png")
            savename = savename.replace(".mp3", "_" + str(counter) + ".png")
            savepath = destination_path + savename

            # If the spectrogram exists, continue.
            if os.path.exists(savepath):
                continue

            # Save the spectrogram as a PNG file.
            logging.info("Writing " + savepath)
            pyplot.imsave(savepath, data, cmap='Greys')
        
        else:
            # Add the data to our dataset.
            dataset.append((data.flatten(), classification))
          
        # Free memory. This is essential to prevent leaks!
        pyplot.close('all')

def get_classification(filename):
    """Look up the classification of this recording."""

    connection = None
    id_number = filename.replace(".mp3", "")
    id_number = id_number.replace(".wav", "")

    try:
        connection = psycopg2.connect(
            database='xeno-canto-data', user='ehrenbrav')
        cursor = connection.cursor()
        cursor.execute("SELECT en FROM recordings WHERE id=%s;", (id_number,))
        classification = cursor.fetchall()
        cursor.close()
        connection.close()
        return classification[0][0]

    except psycopg2.DatabaseError, exception:
        print exception
        sys.exit(1)

    finally:
        if connection:
            connection.close()

if __name__ == '__main__':

    # Get the file paths.
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', type=str)
    parser.add_argument('destination_path', type=str)
    args = parser.parse_args()
    source_path = args.source_path
    destination_path = args.destination_path

    if not destination_path.endswith(os.sep):
        destination_path = destination_path + os.sep

    # Set up logging.
    log_name = datetime.datetime.now().strftime('%y%m%d-%H%M') + ".log"
    logging.basicConfig(filename="../" + log_name, level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s')

    # If the argument is a file, make the spectrogram.
    if os.path.isfile(source_path):
        graph_spectrogram(source_path, destination_path)

    # If the argument is a directory, process all the underling audio files.
    if os.path.isdir(source_path):

        # Create dataset container: a list of tuples of (data, classification).
        dataset = []
        for file_in_dir in os.listdir(source_path):
            file_in_dir_path = os.path.join(source_path, file_in_dir)
            graph_spectrogram(file_in_dir_path, destination_path, dataset)

        # Save the dataset.
        print "Saving data..."
        output = open('dataset.pkl', 'wb')
        cPickle.dump(dataset, output, -1)
        output.close()

    else:
        print "Error: file not found: " + source_path
        exit(1)
