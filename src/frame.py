"""
This is the class for an audio frame.
Typically it is 512 samples.
"""

class Frame():

    FRAME_SIZE = 512

    def __init__(self):
        self.samples = []
