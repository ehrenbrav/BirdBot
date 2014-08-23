#!/usr/bin/env python
"""
This module divides a wav file into euqally sized
spectrograms. In order to increase the amount of training
data available, it first starts from the beginning of the file,
discarding the remainder that doesn't divide equally into the
spectrogram frame size, then starts with an offset equal to the
remainder, to ensure that the entire wav file is covered.
"""

import wav_file_importer as wfi
import logging
import datetime
import argparse
import os
import matplotlib as mpl
import scipy as sp
import psycopg2
import sys
import cPickle
import gzip
import json
from random import shuffle

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')

from matplotlib import pyplot
import numpy as np
import birdbot.params as p

# Write the actual spectrogram?
DRAW_SPECTROGRAM = False

#pylint: disable=R0914,W0621 
def add_audio_to_dataset(
        source_path, destination_path, dataset, classification_map):
    """Generate the spectrograms and optionally save as png files."""

    # Get the name of the wav file.
    filename = os.path.basename(source_path)
    logging.info("Processing " + filename)
    print "Processing " + filename

    # Load the wav file.
    sample_rate, full_audio_data = wfi.validate_and_read_file(source_path)
    frame_size = sample_rate * p.SPECTROGRAM_DURATION
    remainder = len(full_audio_data) % frame_size

    # Check to see we actually have enough audio data.
    if len(full_audio_data) < frame_size:
        logging.error("Audio clip is too short")
        print "Audio clip is too short."
        return

    # Get the classification of this recording.
    classification = get_classification(filename, classification_map)

    # Chop the wav file into equally sized pieces, starting from
    # the front, then do the same starting from the back.
    front_samples = [full_audio_data[i:i+frame_size] \
            for i in range(0, len(full_audio_data) - remainder, frame_size)]

    end_samples = [full_audio_data[i:i+frame_size] \
            for i in range(remainder, len(full_audio_data), frame_size)]

    # Make the spectrograms.
    for counter, sample in enumerate(front_samples + end_samples):

        # Calculate the spectrogram data.
        data = calculate_spectrogram(sample, sample_rate)

        # Draw the actual spectrograms?
        if DRAW_SPECTROGRAM:
            draw_spectrogram(filename, destination_path, data, counter)
            
        # Add the data to our dataset.
        dataset.append((data.flatten(), classification))
          
        # Free memory. This is essential to prevent leaks!
        pyplot.close('all')

def get_classification(filename, classification_map):
    """
    Look up the classification of this recording.
    Then add it to our classification_map.
    """

    connection = None
    id_number = filename.replace(".mp3", "")
    id_number = id_number.replace(".wav", "")

    try:
        connection = psycopg2.connect(
            database='xeno-canto-data', user='ehrenbrav')
        cursor = connection.cursor()
        cursor.execute("SELECT en FROM recordings WHERE id=%s;", (id_number,))
        classification = cursor.fetchall()[0][0]
        cursor.close()

        # Add this to our classification_map if not already there.
        if classification not in classification_map:
            classification_map[classification] = len(classification_map)

        return classification_map[classification]

    except psycopg2.DatabaseError, exception:
        print exception
        sys.exit(1)

    finally:
        if connection:
            connection.close()

def divide_dataset(dataset, classification_map):
    """
    This divides the dataset into
    training, testing, and validation
    data. The data is returned as a tuple:
    (testing_data, validation_data, testing_data).
    Each of these is itself a tuple: (examples, classifications).
    Examples is a ndarray of shape (num_examples, (width x height)).
    Classifications is a ndarray of shape (num_examples).
    """
    #pylint: disable=E1101

    # Shuffle the dataset to ensure we don't have any
    # unintentional biases due to how the examples
    # are indexed.
    shuffle(dataset)
    
    # Get shape of the data (width x height)
    spectrogram_size = dataset[0][0].shape[0]

    # Calculate size of the 3 sets.
    num_examples = len(dataset)
    num_training = int(num_examples * p.PERCENT_TRAINING)
    num_validate = int((num_examples - num_training) / 2)
    num_testing = num_examples - num_training - num_validate

    # Allocate the ndarrays.
    examples = np.zeros((num_examples, spectrogram_size), dtype=np.int8)
    classifications = np.zeros(num_examples, dtype=np.int8)

    # Populate the arrays.
    for index, example in enumerate(dataset):
        examples[index] = example[0]
        classifications[index] = example[1]

    # Split the arrays.
    training_data = (examples[:num_training], classifications[:num_training])
    testing_data = (examples[num_training:(num_training + num_testing)],
                    classifications[num_training:(num_training + num_testing)])
    validation_data = (examples[(num_examples - num_validate):],
                       classifications[(num_examples - num_validate):])

    return (training_data, validation_data, testing_data, classification_map)

def calculate_spectrogram(sample, sample_rate):
    """
    Calculate the spectrogram data given an audio sample.
    """

    #pylint: disable=C0103,E1101,W0612
    
    # Compute the spectrogram.
    Pxx, freqs, bins, im = pyplot.specgram(
        sample, NFFT=1024, Fs=sample_rate, noverlap=512)
    
    # Chop off useless frequencies.
    Pxx = Pxx[(freqs > p.MIN_FREQUENCY) & (freqs < p.MAX_FREQUENCY)]
    
    # Convert to dB scale and flip.
    data = 10. * np.log10(Pxx)
    data = np.flipud(data)

    # Resize to 256 x 256.
    scaled_data = sp.misc.imresize(
        data, (p.SPECTROGRAM_SIDE_SIZE, p.SPECTROGRAM_SIDE_SIZE))

    # Cast data to int8.
    return scaled_data

def draw_spectrogram(filename, destination_path, data, counter):
    """
    Actually draw the spectrograms in destination_path.
    """
    
    # Get add the number of spectrogram sample this is.
    savename = filename
    savename = savename.replace(".wav", "_" + str(counter) + ".png")
    savename = savename.replace(".mp3", "_" + str(counter) + ".png")
    savepath = destination_path + savename
    
    # If the spectrogram exists, continue.
    if os.path.exists(savepath):
        return

    # Save the spectrogram as a PNG file.
    logging.info("Writing " + savepath)
    pyplot.imsave(savepath, data, cmap='Greys')

if __name__ == '__main__':

    #pylint: disable=C0103

    # Get the file paths.
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', type=str)
    parser.add_argument(
        '--outputdir', help="Directory to write the spectrograms.", type=str)
    args = parser.parse_args()
    source_path = args.source_path
    destination_path = ""
    if args.outputdir:
        destination_path = args.outputdir

    if not destination_path.endswith(os.sep):
        destination_path = destination_path + os.sep

    # Set up logging.
    log_name = datetime.datetime.now().strftime('%y%m%d-%H%M') + ".log"
    logging.basicConfig(filename="../" + log_name, level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s')

    # Set up a mapping from classification string -> integer.
    classification_map = {}

    # Process all the underling audio files.
    if os.path.isdir(source_path):

        # Create dataset container: a list of tuples of (data, classification).
        dataset = []
        for file_in_dir in os.listdir(source_path):
            file_in_dir_path = os.path.join(source_path, file_in_dir)
            add_audio_to_dataset(
                file_in_dir_path, destination_path, dataset, classification_map)

        # Divide the dataset into training, validation, and testing data.
        divided_dataset = divide_dataset(dataset, classification_map)

        # Save the dataset.
        print "Saving data..."
        output = gzip.open(p.DATASET_PATH, 'wb')
        cPickle.dump(divided_dataset, output, -1)
        output.close()

        # Save the classification map.
        json.dump(classification_map, open(p.CLASSIFICATION_MAP_PATH, 'w'))

        # Print some stats.
        logging.info("Total examples: " + str(len(dataset)))
        logging.info("Total classes: " + str(len(classification_map)))
        print "Total examples: " + str(len(dataset))
        print "Total classes: " + str(len(classification_map))

    else:
        print "Error: file not found: " + source_path
        exit(1)
