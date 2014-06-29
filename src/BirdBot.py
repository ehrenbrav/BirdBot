#!/usr/bin/env python
"""
This is the main application for BirdBot.
"""

import wav_file_importer
import feature_extractor
import arff_writer
import frame_classifier
import numpy as np
import cPickle

sample_frequency, data = wav_file_importer.validate_and_read_file()

features = feature_extractor.extract(sample_frequency, data)

feature_list = arff_writer.get_feature_list()

frame_count = features[feature_list[0]].shape[0]

frame_classifier = classifier.Classifier()

for frame in range(frame_count):
    frame_features = []

    # Loop through all the features in this frame.
    for feature_name in feature_list:
        feature = features[feature_name]

        # Handle MFCCs.
        if feature_name == "mfcc":
            for counter in range(feature_extractor.NUMBER_MFCCS):
                frame_features.append(feature[frame][counter])
            continue

        # Handle spectral stats.
        if feature_name == "spectral_stats":
            for counter in range(feature_extractor.NUMBER_SPECTRAL_STATS):
                frame_features.append(feature[frame][counter])
            continue

        # Handle single-column data.
        frame_features.append(feature[frame][0])

    # Classify the frame.
    frame_features_array = np.asarray(frame_features)
    frame_classifier.classify_frame(frame_features_array)


print features
#cPickle.dump(features, open("features.obj", "wb"))



