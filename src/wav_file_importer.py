"""Grabs and validates the file from the command line"""
import argparse
import subprocess
from scipy.io import wavfile
import numpy as np

def validate_and_read_file(file_path=None):
    
    if file_path is None:
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
    print "File Name: " + file_path
    print "Sample Frequency: " + str(sample_frequency)
    duration = float(data.shape[0]) / float(sample_frequency)
    print "Duration (s): " + str(round(duration, 3))
    print "Channels (we only extract the first): " + str(len(data.shape))

    # If there are two columns of data (stereo), only keep one.
    if len(data.shape) > 1:
        data = np.delete(data, 1, 1)
        data = data.flatten()
        
    # Return sample_frequency, data
    return sample_frequency, data
