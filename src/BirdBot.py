#!/usr/bin/env python
"""This is the main method of BirdBot."""

import wav_file_importer
import segmenter
import feature_extractor

sample_frequency, data = wav_file_importer.validate_and_read_file()

frames = segmenter.divide_audio_into_frames(data)

# TODO fix this
for i in range(3):
    feature_extractor.extract_features(frames[i])
