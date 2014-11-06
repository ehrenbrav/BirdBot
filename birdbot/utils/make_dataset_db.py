#!/usr/bin/env python
"""
This module divides a wav file into equally sized
spectrograms, offset by a specified stride distance.
It then saves these to a PostgreSQL database.
"""

import birdbot.utils.wav_file_importer as wfi
import birdbot.utils.create_database as db
import argparse
import os
import matplotlib as mpl
import scipy as sp
import psycopg2
import sys
from random import shuffle

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')

from matplotlib import pyplot
import numpy as np
import birdbot.params as p

#pylint: disable=R0914,W0621

def add_audio_to_dataset(source_path, connection, cursor):
    """Generate the spectrograms."""

    # Get the name of the wav file.
    filename = os.path.basename(source_path)

    # Load the wav file.
    sample_rate, full_audio_data = wfi.validate_and_read_file(source_path)

    # Handle errors.
    if sample_rate == None or full_audio_data == None:
        return

    # Calculate frame size.
    frame_size = sample_rate * p.SPECTROGRAM_DURATION

    # Check to see we actually have enough audio data.
    if len(full_audio_data) < frame_size:
        print "Audio clip is too short."
        return

    # Get the classification of this recording.
    classification, id_number = get_classification(filename)

    # Chop the wav file into equally sized pieces,
    # separated by the spectrogram stride distance.
    last_index = len(full_audio_data) - frame_size
    samples = []
    frame_num = 0
    while frame_num < last_index:
        new_sample = full_audio_data[frame_num:(frame_num + frame_size)]
        samples.append(new_sample)
        frame_num = frame_num + (p.SPECTROGRAM_STRIDE * sample_rate)

    # Make the spectrograms.
    for sample in samples:

        # Calculate the spectrogram data.
        data = calculate_spectrogram(sample, sample_rate)

        # Convert the data to a plain-vanilla python list for the db.
        data_list = list(data.flatten().astype(int))

        # Add the data to our dataset.
        cursor.execute(
            "INSERT INTO spectrograms (data, classification, recording_id) VALUES (%s, %s, %s)", (data_list, classification, id_number))
        connection.commit()

        # Free memory. This is essential to prevent leaks!
        pyplot.close('all')

def get_classification(filename):
    """
    Look up the classification of this recording.
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

        return classification, id_number

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

    # Resize to specified size.
    scaled_data = sp.misc.imresize(
        data, (p.SPECTROGRAM_SIDE_SIZE, p.SPECTROGRAM_SIDE_SIZE))

    # Cast data to int8.
    return scaled_data

if __name__ == '__main__':

    #pylint: disable=C0103

    # Get the file paths.
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', type=str)
    args = parser.parse_args()
    source_path = args.source_path

    # See if the database already exists. If so, quit. If not, create.
    connection = None
    try:
        connection = psycopg2.connect(database=p.DATABASE, user='ehrenbrav')
        print "Database already exists...quitting."
        exit(1)

    except psycopg2.OperationalError, exception:
        print "Creating database..."
        db.create(p.DATABASE)

    finally:
        if connection:
            connection.close()

    # Process all the underling audio files.
    if os.path.isdir(source_path):

        # Get file count.
        file_count = len(os.listdir(source_path))

        # Set up the database.
        try:
            connection = psycopg2.connect(database=p.DATABASE, user='ehrenbrav')
            cursor = connection.cursor()

            # Create dataset: a list of tuples of (data, classification).
            # TODO check if the file is already in the database.
            counter = 1
            for file_in_dir in os.listdir(source_path):
                file_in_dir_path = os.path.join(source_path, file_in_dir)
                add_audio_to_dataset(file_in_dir_path, connection, cursor)
                print "File " + str(counter) + " of " + str(file_count)
                counter += 1

        except psycopg2.DatabaseError, exception:
            print exception
            exit(1)

        finally:
            if connection:
                connection.close()

    else:
        print "Error: file not found: " + source_path
        exit(1)
