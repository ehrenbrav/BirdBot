#!/usr/bin/env python

import wav_file_importer 
import yaafe_feature_extractor
import arff
import cPickle

sample_frequency, data = wav_file_importer.validate_and_read_file()

features = yaafe_feature_extractor.extract(sample_frequency, data)

arff_file = arff.dumps(features)

print features
#cPickle.dump(features, open("features.obj", "wb"))

