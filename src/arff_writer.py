#!/usr/bin/env python
"""
This module takes a directory
of directories each of which contain
sound files, and uses the
current settings of the feature extractor
to write an ARFF file. The name of each directory
must be the same as the bird whose sounds
are recorded in the files within that directory.
"""

import argparse
import os
import feature_extractor
import wav_file_importer

# Name of ARFF file.
ARFF_NAME = 'training_data.arff'

# Get name of directory specified on the command line.
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
path = parser.parse_args().path

# Handle error of no directory provided.
if not os.path.isdir(path):
    print """
    The path specified is not a directory.
    The training data must be in a series
    of directories, each within the directory
    named on the command line when invoking
    this utility.
    """
    exit(1)

# Get the list of sub-directories.
directory_list = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) ]

with open(ARFF_NAME, 'w') as arff_file:

    arff_file.write("@RELATION bird_calls\n")

    # Get the list of the features we'll be extracting.
    engine = feature_extractor.configure_engine(44100)
    feature_list = engine.getOutputs().keys()

    # Write time attribute.
    arff_file.write("@ATTRIBUTE frame_number NUMERIC\n")

    # Write attributes.
    for feature in feature_list:

        # Hanlde multiple MFCC coefficients.
        if feature == "mfcc":
            for counter in range(feature_extractor.NUMBER_MFCCS):
                arff_file.write("@ATTRIBUTE mfcc" + str(counter) + " NUMERIC\n")
            continue

        # Handle multiple spectral statistics.
        if feature == "spectral_stats":
            for counter in range(feature_extractor.NUMBER_SPECTRAL_STATS):
                arff_file.write("@ATTRIBUTE spectral_stats" + str(counter) + " NUMERIC\n")
            continue

        # Write a normal attribute.
        arff_file.write("@ATTRIBUTE " + feature + " NUMERIC\n")

    # Write classes.
    class_list = ""
    for class_name in directory_list:
        class_list += class_name + ","

    class_list = class_list[:-1]
    arff_file.write("@ATTRIBUTE class {" + class_list + "}\n")

    # Finish up.
    arff_file.write("@DATA\n")
    
    # Loop through all files and write into a single ARFF.
    for root, dirs, files in os.walk(path):

        # Skip the root directory.
        if root == path:
            continue
            
        # The class name is the name of the subdirectory.
        class_name = os.path.split(root)[1]
        
        for file_name in files:
            sample_frequency, data = wav_file_importer.validate_and_read_file(root + os.sep + file_name)
            features = feature_extractor.extract(sample_frequency, data)

            # Figure out how many frames there are.
            frame_count = features[feature_list[0]].shape[0]
            
            # Write the data. Ensure this is in the same order as 
            # the attributes listed at the top of the file.
            for frame in range(frame_count):

                # Write the frame number.
                arff_file.write(str(frame) + ",")
                
                # Loop through all features for this frame.
                for feature_name in feature_list:
                    feature = features[feature_name]

                    # Handle MFCCs.
                    if feature_name == "mfcc":
                        for counter in range(feature_extractor.NUMBER_MFCCS):
                            arff_file.write(str(feature[frame][counter]) + ",")
                        continue

                    # Handle spectral stats.
                    if feature_name == "spectral_stats":
                        for counter in range(feature_extractor.NUMBER_SPECTRAL_STATS):
                            arff_file.write(str(feature[frame][counter]) + ",")
                        continue

                    # Handle the single-column data.
                    arff_file.write(str(feature[frame][0]) + ",")
                
                # Now write the class.
                arff_file.write(class_name + "\n")
