#!/usr/bin/env python

"""
Ping the Xeno-Canto API.
"""

import psycopg2
import requests
import sys

XENO_CANTO_URL = \
'http://www.xeno-canto.org/api/2/recordings?query=+cnt%3A"United+States"'

def parse():
    """Grab all the results and add them to the database."""

    # First, figure out how many pages of data there are.
    num_pages = get_response(XENO_CANTO_URL).json['numPages']

    # Now, process each page of data.
    recordings = []
    for page in range(1, num_pages + 1):
        print "Getting page " + str(page)
        url = XENO_CANTO_URL + '&page=' + str(page)
        web_data = get_response(url).json['recordings']


        for entry in web_data:
            recordings.append(entry)

    # Now, write the data to the database.
    connection = None

    try:
        connection = psycopg2.connect(
            database = 'xeno-canto-data', user='ehrenbrav')
        cursor = connection.cursor()

        for recording in recordings:

            # Check to see if the entry already exists.
            cursor.execute(
                "SELECT * FROM recordings WHERE id=%(id)s", recording)

            # If not, insert.
            if cursor.fetchone() == None:
                cursor.execute(
                    """INSERT INTO recordings VALUES (
                    %(loc)s,
                    %(cnt)s,
                    %(en)s,
                    %(lic)s,
                    %(url)s,
                    %(sp)s,
                    %(ssp)s,
                    %(gen)s,
                    %(file)s,
                    %(lat)s,
                    %(rec)s,
                    %(lng)s,
                    %(type)s,
                    %(id)s);""", recording)
                print "Writing recording: " + recording['id']
        connection.commit()

    except psycopg2.DatabaseError, exception:
        if connection:
            connection.rollback()
        print exception
        sys.exit(1)

    finally:
        if connection:
            connection.close()

def get_response(url):
    """Use requests module to grab the HTML."""
    try:
        response_from_website = requests.get(url)

        if response_from_website.status_code != 200:
            print("Didn't get 200 http status code: %s",
                  response_from_website.status_code)
            exit(1)

    except requests.exceptions.RequestException, e:
        print("Connection Error: %s", e)
        exit(1)

    return response_from_website

if __name__ == "__main__":
    parse()
