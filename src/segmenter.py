"""
This module reads a .wav file
and divides it into individual
frames of SAMPLES_PER_FRAME each.
"""
from frame import Frame

def divide_audio_into_frames(data):
    """Breaks audio samples into individual frames."""

    # Break samples into equally-sized frames.
    counter = 0
    frame = Frame()
    frames = []
    sample = 0

    while sample < len(data):

        # Frames have 50% overlap.
        # We discard the last frame.
        if counter == Frame.FRAME_SIZE:
            counter = 0
            sample = sample - Frame.FRAME_SIZE / 2
            frames.append(frame)
            frame = Frame()
        frame.samples.append(data[sample])
        counter += 1
        sample += 1
    return frames

if __name__ == "__main__":
    divide_audio_into_frames()



