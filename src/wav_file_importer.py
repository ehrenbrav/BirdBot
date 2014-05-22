"""Grabs and validates the file from the command line"""
import argparse
import subprocess
from scipy.io import wavfile
from math import sqrt
from numpy import mean

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

    # Read the wav file.
    sample_frequency, data = wavfile.read(file_path)

    # Print some info about the file.
    print "Sample Frequency: " + str(sample_frequency)
    duration = float(data.shape[0]) / float(sample_frequency)
    print "Duration (s): " + str(round(duration, 3))
    print "Channels (we only extract the first): " + str(len(data.shape))
    print "RMS: " + str(sqrt(mean(data**2)))

    # Return sample_frequency, data
    return sample_frequency, data
