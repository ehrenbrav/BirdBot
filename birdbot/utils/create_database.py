"""
This script creates a database to hold the
spectrograms.
"""

import subprocess
import psycopg2

def create(database_name):
    """Create the database."""

    # Make the actual database.
    subprocess.call(["createdb", database_name])

    # Create the table of spectrograms.
    connection = None
    try:
        connection = psycopg2.connect(database=database_name, user='ehrenbrav')
        cursor = connection.cursor()
        cursor.execute(
            """CREATE TABLE spectrograms (
            id serial PRIMARY KEY,
            data integer[],
            classification text,
            recording_id text,
            dataset_category text,
            classification_id integer);""")
        connection.commit()

    except psycopg2.DatabaseError, exception:
        if connection:
            connection.rollback()
            print exception
            exit(1)

    finally:
        if connection:
            connection.close()
