#!/usr/bin/env python

import wav_file_importer 
import yaafe_feature_extractor

sample_frequency, data = wav_file_importer.validate_and_read_file()

features = yaafe_feature_extractor.extract(sample_frequency, data)

print features
