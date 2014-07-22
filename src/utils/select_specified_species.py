#!/usr/bin/env python
"""
This script allows you to enter
a species name as an argument.
It will then copy all recordings
of that bird to a directory.
"""

#pylint: disable=C0103

import argparse
import psycopg2
import sys
import os.path
import shutil

DESTINATION_DIRECTORY = "../../training_data/test/"
SOURCE_DIRECTORY = "../../training_data/audio/"

parser = argparse.ArgumentParser()
parser.add_argument('species', type=str)
species = parser.parse_args().species

connection = None

try:
    connection = psycopg2.connect(
        database='xeno-canto-data', user='ehrenbrav')
    cursor = connection.cursor()

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

