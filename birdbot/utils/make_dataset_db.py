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
import random

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')

from matplotlib import pyplot
import numpy as np
import birdbot.params as p

#pylint: disable=R0914,W0621

def add_audio_to_dataset(source_path, connection, cursor, classification_map):
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

    # Get the classification and the Xeno-Canto id of this recording.
    classification, recording_id = get_classification(filename)

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

        # Assign to a category: training, testing, or validation.
        random_number = random.random()
        dataset_category = None
        if random_number <= p.PERCENT_TRAINING:
            dataset_category = "train"
        elif random_number < (1. - ((1. - p.PERCENT_TRAINING) / 2)):
            dataset_category = "test"
        else:
            dataset_category = "valid"

        # See if there's a number assigned to this class.
        if classification not in classification_map:
            num_keys = len(classification_map.keys())
            classification_map[classification] = num_keys

        # Add the data to our dataset.
        cursor.execute(
            """INSERT INTO spectrograms
            (data,
            classification,
            recording_id,
            dataset_category,
            classification_id)
            VALUES (%s, %s, %s, %s, %s);""",
            (data_list,
             classification,
             recording_id,
             dataset_category,
             classification_map[classification]))
        connection.commit()

        # Free memory. This is essential to prevent leaks!
        pyplot.close('all')

def get_classification(filename):
    """
    Look up the classification of this recording.
    """

    connection = None
    recording_id = filename.replace(".mp3", "")
    recording_id = recording_id.replace(".wav", "")

    try:
        connection = psycopg2.connect(
            database='xeno-canto-data', user='ehrenbrav')
        cursor = connection.cursor()
        cursor.execute(
            "SELECT en FROM recordings WHERE id=%s;", (recording_id,))
        classification = cursor.fetchall()[0][0]
        cursor.close()

        return classification, recording_id

    except psycopg2.DatabaseError, exception:
        print exception
        sys.exit(1)

    finally:
        if connection:
            connection.close()

def data_already_exists_in_db(path, processed_files):
    """Check if the file has already been processed and
    is in the DB."""

    # Get the id from the filename.
    filename = os.path.basename(path)
    recording_id = filename.replace(".mp3", "")
    recording_id = recording_id.replace(".wav", "")

    # Check if the current file is already there.
    if (recording_id,) in processed_files:
        return True
    return False

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

    # See if the database already exists. If not, create.
    connection = None
    try:
        connection = psycopg2.connect(database=p.DATABASE, user='ehrenbrav')
        print "Database already exists..."

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

            # Get list of already processed files (if any).
            cursor.execute("SELECT recording_ID FROM spectrograms;")
            processed_files = cursor.fetchall()

            # Create dataset: a list of tuples of (data, classification).
            counter = 1
            classification_map = {}
            for file_in_dir in os.listdir(source_path):

                # Keep track of the statistics.
                print "File " + str(counter) + " of " + str(file_count)
                counter += 1

                # If the file has already been processed and is in the DB, skip.
                file_in_dir_path = os.path.join(source_path, file_in_dir)
                if data_already_exists_in_db(file_in_dir_path, processed_files):
                    print "File has already been processed."
                    continue

                # Otherwise, add to the dataset.
                add_audio_to_dataset(
                    file_in_dir_path, connection, cursor, classification_map)

        except psycopg2.DatabaseError, exception:
            print exception
            exit(1)

        finally:
            if connection:
                connection.close()

    else:
        print "Error: file not found: " + source_path
        exit(1)
