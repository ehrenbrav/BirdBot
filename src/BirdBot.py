#!/usr/bin/env python
"""This is the main method of BirdBot."""

import wav_file_importer
import segmenter
import feature_extractor
import pylab as plt
import numpy as np

sample_frequency, data = wav_file_importer.validate_and_read_file()
frames = segmenter.divide_audio_into_frames(sample_frequency, data)

for i in range(len(frames)):
    feature_extractor.extract_features(frames[i])
#    frames[i].graph_frame()
    
mean_trajectory = []

for frame in frames:
    mean_trajectory.append(frame.mean)

frame_number = np.asarray(range(len(frames)))
plt.plot(frame_number, np.asarray(mean_trajectory))
plt.show()
