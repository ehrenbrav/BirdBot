#!/usr/bin/env python
"""This is the main method of BirdBot."""

import wav_file_importer
import segmenter
import feature_generator

sample_frequency, data = wav_file_importer.validate_and_read_file()

frames = segmenter.divide_audio_into_frames(data[0])

for frame in frames:
    feature_generator.generate_features(frame)
