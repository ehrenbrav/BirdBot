"""Grabs and validates the file from the command line"""
import subprocess
import logging
import os.path
from scipy.io import wavfile
import numpy as np

def validate_and_read_file(file_path=None):
    """Get the data out of a wav or mp3 file."""

    # Check if file exists.
    if not os.path.isfile(file_path):
        print "Error: no file found at " + file_path
        exit(1)

    # Handle various audio formats.
    file_command = subprocess.Popen(
        ['file', file_path], stdout=subprocess.PIPE)
    output = file_command.stdout.read()

    if 'ID3' in output:
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
    """Use avconv to get the audio data."""
    sample_rate = 44100

    # Check whether we have ffmpeg or avconv.
    if os.path.isfile("/usr/bin/avconv"):
        mp3_extractor = 'avconv'
    elif os.path.isfile("/usr/bin/ffmpeg"):
        mp3_extractor = 'ffmpeg'
    else:
        print """Error: no mp3 extractor found.
                 We require either ffmpeg or avconv
                 and expect these to be in the /usr/bin
                 directory."""
        exit(1)

    command = [mp3_extractor,           # Path to native binary.
               '-i', path,              # Location of mp3.
               '-f', 's16le',           # Format.
               '-acodec', 'pcm_s16le',  # Get raw 16-bit output.
               '-ar', str(sample_rate), # Sample rate.
               '-ac', '1',              # Mono.
               '-loglevel', 'quiet',    # Limit output.
               '-' ]
    pipe = subprocess.Popen(
        command, stdout=subprocess.PIPE, bufsize=10**8, close_fds=True)
    raw_audio = pipe.stdout.read()
    data = np.fromstring(raw_audio, dtype='int16')
    return sample_rate, data
