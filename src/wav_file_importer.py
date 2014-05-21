"""Grabs and validates the file from the command line"""
import argparse
import subprocess
from scipy.io import wavfile

def validate_and_read_file():
    """Grab the file path."""
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    file_path = parser.parse_args().file_path
    file_command = subprocess.Popen(['file', file_path], stdout=subprocess.PIPE)
    output = file_command.stdout.read()
    if 'WAVE' not in output:
        print "Error: " \
        + file_path \
        + " does not appear to be a .wav file"
        exit(1)

    # Return sample_frequency, data
    return wavfile.read(file_path)
