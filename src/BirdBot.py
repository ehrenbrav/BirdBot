#!/usr/bin/env python
"""
This is the main application for BirdBot.
"""

import wav_file_importer
import feature_extractor
import cPickle

sample_frequency, data = wav_file_importer.validate_and_read_file()

features = feature_extractor.extract(sample_frequency, data)

print features
#cPickle.dump(features, open("features.obj", "wb"))



