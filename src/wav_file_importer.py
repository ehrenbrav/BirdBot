"""Grabs and validates the file from the command line"""
import subprocess
import logging
from scipy.io import wavfile
import numpy as np

def validate_and_read_file(file_path=None):
    """Get the data out of a wav or mp3 file."""

    # Handle various audio formats.
    file_command = subprocess.Popen(
        ['file', file_path], stdout=subprocess.PIPE)
    output = file_command.stdout.read()

    if 'MPEG' in output:
        sample_frequency, data = extract_mp3(file_path)
    elif 'WAVE' in output:
        sample_frequency, data = wavfile.read(file_path)
    else:
        logging.error("Error: " \
            + file_path \
            + " does not appear to be a .wav file")
        exit(1)

    # Print some info about the file.
    logging.info("File Name: " + file_path)
    logging.info("Sample Frequency: " + str(sample_frequency))
    duration = float(data.shape[0]) / float(sample_frequency)
    logging.info("Duration (s): " + str(round(duration, 3)))
    logging.info("Channels (we only extract the first): " + str(len(data.shape)))

    # If there are two columns of data (stereo), only keep one.
    if len(data.shape) > 1:
        data = np.delete(data, 1, 1)
        data = data.flatten()

    # Return sample_frequency, data
    return sample_frequency, data

def extract_mp3(path):
    """Use ffmpeg to get the audio data."""
    sample_rate = 44100

    command = ['ffmpeg',                # Path to native binary.
               '-i', path,              # Location of mp3.
               '-f', 's16le',           # Format.
               '-acodec', 'pcm_s16le',  # Get raw 16-bit output.
               '-ar', str(sample_rate), # Sample rate.
               '-ac', '1',              # Mono.
               '-loglevel', 'quiet',    # Limit output.
               '-' ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    raw_audio = pipe.stdout.read()
    data = np.fromstring(raw_audio, dtype='int16')
    return sample_rate, data
