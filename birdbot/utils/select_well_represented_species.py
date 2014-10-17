#!/usr/bin/env python
"""
This script creates a dataset of species
where at least X number of recordings exist.
"""

#pylint: disable=C0103

import collections
import psycopg2
import sys
import os.path
import shutil

DESTINATION_DIRECTORY = "../../training_data/50_plus_recordings/"
SOURCE_DIRECTORY = "../../training_data/audio/"

MIN_NUMBER_OF_EXAMPLES = 50

connection = None

try:

    connection = psycopg2.connect(
        database='xeno-canto-data', user='ehrenbrav')
    cursor = connection.cursor()

    # Make a list of all the species.
    cursor.execute("SELECT en FROM recordings")
    species_list = cursor.fetchall()

    counter = collections.Counter(species_list)
    species_to_fetch = []
    for species in counter.items():
        if species[1] > MIN_NUMBER_OF_EXAMPLES:
            species_to_fetch.append(species[0])

    for species in species_to_fetch:
        cursor.execute("SELECT id FROM recordings WHERE en=%s;", (species,))
        id_numbers = cursor.fetchall()

        for id_number in id_numbers:
            file_type = ".mp3"
            source_path = SOURCE_DIRECTORY + str(id_number[0]) + ".mp3"
            print source_path
            if not os.path.isfile(source_path):
                source_path = SOURCE_DIRECTORY + str(id_number[0]) + ".wav"
                file_type = ".wav"
            if not os.path.isfile(source_path):
                print "Error: file not found - " + source_path
                sys.exit(1)

            destination_path = DESTINATION_DIRECTORY + str(id_number[0]) + file_type
            shutil.copy(source_path, destination_path)

except psycopg2.DatabaseError, exception:
    print exception
    sys.exit(1)

finally:
    if connection:
        connection.close()

