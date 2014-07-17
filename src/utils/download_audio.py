#!/usr/bin/env python
"""
This script downloads *all* the audio data (in
mostly mp3 format) that is specified in the meta-data
grabbed from the Xeno-Canto website.
"""

import sys
import os
import psycopg2
import requests

# Location of training data.
TRAINING_DATA_PATH = "../../training_data/audio/"

def download_xc_audio():
    """Get *all* audio data."""

    # Generate list of URLs indexed by the id number.
    connection = None

    try:
        connection = psycopg2.connect(
            database='xeno-canto-data', user='ehrenbrav')
        cursor = connection.cursor()

        cursor.execute(
            "SELECT id, file FROM recordings")
        urls = cursor.fetchall()

        # Loop through all the URLs and download.
        for url in urls:
            path = TRAINING_DATA_PATH + str(url[0]) + ".mp3"

            # See if it already exists.
            if os.path.isfile(path):
                continue

            # Download the file.
            response = requests.get(url[1])
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            print "Downloading " + path

    except psycopg2.DatabaseError, exception:
        if connection:
            connection.rollback()
        print exception
        sys.exit(1)

    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    download_xc_audio()
