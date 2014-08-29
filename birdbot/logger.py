"""Simple class to handle logging."""

import logging

# pylint: disable=R0903

class Logger(object):
    """Sets up global logging."""

    def __init__(self):
        """Sets up global logging."""
                # Logging.
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        short_formatter = logging.Formatter('%(message)s')

        # Print to console.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(short_formatter)
        logger.addHandler(console_handler)

        # Print to file.
        file_handler = logging.FileHandler('../birdbot.log')
        long_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(long_formatter)
        logger.addHandler(file_handler)

